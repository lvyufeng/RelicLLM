"""A split-K prototype for the decode sparse attention, and what it would buy.

The shipped kernel launches `grid = (batch * query_len, kv_heads)`, which for a batch-1 decode
step with one KV head a rank is **one CTA**, and `kMmaKvTile = 32` against a 2048 position budget
means that one CTA walks 64 tiles in sequence. One SM of 128 does the work, which is why the
kernel measures 294 us for about 6.3 MFLOP.

This prototype answers the obvious question before anyone writes production CUDA for it: if the
same arithmetic is split across many CTAs and combined, how much faster is it, and does it still
produce the same answer? It is deliberately a *prototype*: it lives here rather than in
`relic-core` because the kernel belongs to that repository, and it is compiled through
`torch.utils.cpp_extension` so nothing in the installed extension is touched.

Run on a leased CUDA host, from the repository root:
  python tests/probe_qwen4_exp_qsa_splitk.py

It runs on one card: the question is a per-launch kernel time on a single device, and a
single process also means the prototype is compiled once instead of once per rank.

The comparison is against `qwen4_exp_qsa_bf16_forward`, the op the model actually calls.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kDim = 256;
constexpr int kHeadsPerKv = 3;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
#ifndef QSA_SPLITK_SPLITS
#define QSA_SPLITK_SPLITS 16
#endif
constexpr int kSplits = QSA_SPLITK_SPLITS;
constexpr int kMaxChunk = 256;   // 2048 positions over 16 splits is 128, so this is slack

__device__ __forceinline__ float ld(const __nv_bfloat16* p) {
    return __bfloat162float(*p);
}

// One CTA per (split, kv_head): a slice of the selected positions reduced to a running max, a
// running sum and a partial output. This is flash-decoding's first pass.
//
// Every warp walks the same number of iterations (`kMaxChunk / kWarps` is 8, and the loop bound is
// the chunk rounded up to a multiple of the warp count) so that the block-wide `__syncthreads`
// after scoring cannot be reached by some warps and not others.
__global__ void qsa_splitk_partial_kernel(
        const __nv_bfloat16* __restrict__ query,   // [1, 1, heads, kDim]
        const __nv_bfloat16* __restrict__ key,     // [1, kv_heads, kv_len, kDim]
        const __nv_bfloat16* __restrict__ value,
        const int32_t* __restrict__ selected,      // [1, 1, selected_len]
        float* __restrict__ part_acc,              // [kv_heads, splits, heads, kDim]
        float* __restrict__ part_m,                // [kv_heads, splits, heads]
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

    __shared__ float q_sh[kHeadsPerKv][kDim];
    __shared__ float scores[kHeadsPerKv][kMaxChunk];
    __shared__ int32_t ids[kMaxChunk];

    const int first_head = kv_head * kHeadsPerKv;
    const __nv_bfloat16* q_base = query + static_cast<int64_t>(first_head) * kDim;
    for (int i = tid; i < kHeadsPerKv * kDim; i += kThreads) {
        q_sh[i / kDim][i % kDim] = ld(q_base + i);
    }
    for (int i = tid; i < count; i += kThreads) {
        ids[i] = selected[begin + i];
    }
    __syncthreads();

    const int64_t kv_head_base = static_cast<int64_t>(kv_head) * kv_len * kDim;

    // One warp a position: lane-parallel dot products, one per head of this kv head.
    for (int base = 0; base < count; base += kWarps) {
        const int slot = base + warp;
        float acc[kHeadsPerKv] = {0.0f, 0.0f, 0.0f};
        const int token = slot < count ? ids[slot] : -1;
        if (token >= 0 && token < kv_len) {
            const __nv_bfloat16* kp =
                key + kv_head_base + static_cast<int64_t>(token) * kDim;
            for (int d = lane; d < kDim; d += 32) {
                const float kv = ld(kp + d);
#pragma unroll
                for (int h = 0; h < kHeadsPerKv; ++h) {
                    acc[h] += q_sh[h][d] * kv;
                }
            }
        }
#pragma unroll
        for (int h = 0; h < kHeadsPerKv; ++h) {
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

    // Softmax over this slice, then the value accumulation.
    //
    // This has to be three separated phases. The obvious one-pass version -- every thread scans
    // `scores` for the maximum and then overwrites `scores` with the exponentials in the same loop
    // body -- is a race: a thread that has already written `exp(score - max)` into a slot leaves a
    // value of at most 1 there, and a thread still scanning for the maximum reads it and gets a
    // maximum that depends on how far its siblings have progressed. The result is correlated with
    // the right answer but wrong, which is exactly what the first version of this measured.
    float m[kHeadsPerKv];
    float l[kHeadsPerKv];
    float out[kHeadsPerKv];
    for (int h = 0; h < kHeadsPerKv; ++h) {
        m[h] = -INFINITY;
        for (int slot = 0; slot < count; ++slot) {
            m[h] = fmaxf(m[h], scores[h][slot]);
        }
    }
    __syncthreads();   // every thread is done reading the scores before any of them is rewritten

    for (int h = 0; h < kHeadsPerKv; ++h) {
        for (int slot = tid; slot < count; slot += kThreads) {
            scores[h][slot] = scores[h][slot] == -INFINITY
                ? 0.0f
                : __expf(scores[h][slot] - m[h]);
        }
    }
    __syncthreads();   // all probabilities are in place

    for (int h = 0; h < kHeadsPerKv; ++h) {
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
        for (int h = 0; h < kHeadsPerKv; ++h) {
            out[h] += scores[h][slot] * vd;
        }
    }

    const int64_t split_index = static_cast<int64_t>(kv_head) * kSplits + split;
#pragma unroll
    for (int h = 0; h < kHeadsPerKv; ++h) {
        part_acc[split_index * kHeadsPerKv * kDim + static_cast<int64_t>(h) * kDim + tid] = out[h];
    }
    if (tid < kHeadsPerKv) {
        part_m[split_index * kHeadsPerKv + tid] = m[tid];
        part_l[split_index * kHeadsPerKv + tid] = l[tid];
    }
}

// flash-decoding's second pass: combine the splits with the max and sum each of them carries.
__global__ void qsa_splitk_combine_kernel(
        const float* __restrict__ part_acc,
        const float* __restrict__ part_m,
        const float* __restrict__ part_l,
        __nv_bfloat16* __restrict__ output,        // [1, 1, heads, kDim]
        int kv_heads) {
    const int kv_head = static_cast<int>(blockIdx.x);
    const int head = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);

    float m = -INFINITY;
    for (int split = 0; split < kSplits; ++split) {
        m = fmaxf(m, part_m[(kv_head * kSplits + split) * kHeadsPerKv + head]);
    }
    float acc = 0.0f;
    float denom = 0.0f;
    for (int split = 0; split < kSplits; ++split) {
        const int64_t idx = static_cast<int64_t>(kv_head) * kSplits + split;
        const float p = part_l[idx * kHeadsPerKv + head];
        const float weight = p == 0.0f ? 0.0f : __expf(part_m[idx * kHeadsPerKv + head] - m);
        denom += p * weight;
        acc += part_acc[idx * kHeadsPerKv * kDim + head * kDim + tid] * weight;
    }
    acc = denom == 0.0f ? 0.0f : acc / denom;
    output[static_cast<int64_t>(kv_head * kHeadsPerKv + head) * kDim + tid] =
        __float2bfloat16(acc);
    (void)kv_heads;
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
    const int selected_len = static_cast<int>(selected.size(2));
    TORCH_CHECK(query.size(2) == kv_heads * kHeadsPerKv,
                "this prototype is specialised for three query heads a kv head");
    TORCH_CHECK(query.size(3) == kDim, "head dim must be 256");
    TORCH_CHECK(selected_len <= kSplits * kMaxChunk, "selection too long for this prototype");

    auto output = torch::empty(query.sizes(), query.options());
    auto fopts = query.options().dtype(torch::kFloat32);
    auto part_acc = torch::empty({kv_heads, kSplits, kHeadsPerKv, kDim}, fopts);
    auto part_m = torch::empty({kv_heads, kSplits, kHeadsPerKv}, fopts);
    auto part_l = torch::empty({kv_heads, kSplits, kHeadsPerKv}, fopts);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const dim3 grid(kSplits, kv_heads);
    qsa_splitk_partial_kernel<<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(query.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(key.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(value.data_ptr()),
        selected.data_ptr<int32_t>(),
        part_acc.data_ptr<float>(),
        part_m.data_ptr<float>(),
        part_l.data_ptr<float>(),
        kv_len, selected_len,
        static_cast<float>(softmax_scale));
    const dim3 grid2(kv_heads, kHeadsPerKv);
    qsa_splitk_combine_kernel<<<grid2, kDim, 0, stream>>>(
        part_acc.data_ptr<float>(),
        part_m.data_ptr<float>(),
        part_l.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        kv_heads);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qsa_splitk_forward", &qsa_splitk_forward, "split-K decode sparse attention");
}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-len", type=int, default=2048)
    parser.add_argument("--kv-len", type=int, default=8192)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--splits", type=int, default=16,
                        help="split count; 1 disables the combine pass for bisecting")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("this probe needs a CUDA host")
    from torch.utils.cpp_extension import load_inline

    report = {"kind": "qwen4_exp_qsa_splitk_probe", "selected_len": args.selected_len,
              "kv_len": args.kv_len, "kv_heads": 1, "heads_per_kv": 3,
              "splits": args.splits}

    ext = load_inline(
        name=f"qwen4_qsa_splitk_probe_s{args.splits}",
        cpp_sources="",
        cuda_sources=_CUDA_SOURCE,
        functions=None,
        extra_cuda_cflags=["-O3", "--use_fast_math",
                           f"-DQSA_SPLITK_SPLITS={args.splits}"],
        verbose=False,
    )
    report["compiled"] = True

    from relicllm.models.qwen4_exp.attention import load_ops

    shipped = load_ops()
    if shipped is None or not hasattr(shipped, "qwen4_exp_qsa_bf16_forward"):
        raise RuntimeError("the shipped qwen4_exp_qsa_bf16_forward op is not available")

    torch.manual_seed(0)
    device = torch.device("cuda:0")
    heads, kv_heads, dim = 3, 1, args.head_dim
    query = torch.randn(1, 1, heads, dim, device=device, dtype=torch.bfloat16)
    key = torch.randn(1, kv_heads, args.kv_len, dim, device=device, dtype=torch.bfloat16)
    value = torch.randn(1, kv_heads, args.kv_len, dim, device=device, dtype=torch.bfloat16)
    # Sorted unique positions, the shape the block indexer produces; then a few invalid slots,
    # which the kernel must exclude rather than read.
    chosen = torch.randperm(args.kv_len, device=device)[: args.selected_len - 3].sort().values
    selected = torch.cat([
        chosen, torch.full((3,), -1, device=device, dtype=torch.int64)
    ]).to(torch.int32).view(1, 1, -1)
    scale = 1.0 / (dim ** 0.5)

    # A plain torch reference of the same arithmetic, so a mismatch can be attributed: whichever
    # op disagrees with it is the one that is wrong.
    def torch_reference(sel: torch.Tensor) -> torch.Tensor:
        idx = sel.view(-1).to(torch.int64)
        valid = (idx >= 0) & (idx < args.kv_len)
        gathered_k = key[0, 0, idx.clamp(min=0)].float()      # [sel, dim]
        gathered_v = value[0, 0, idx.clamp(min=0)].float()
        scores = query[0, 0].float() @ gathered_k.t() * scale  # [heads, sel]
        scores = scores.masked_fill(~valid.unsqueeze(0), float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return (weights @ gathered_v).to(torch.bfloat16).unsqueeze(0).unsqueeze(0)

    ref = torch_reference(selected)
    want = shipped.qwen4_exp_qsa_bf16_forward(query, key, value, selected, scale)
    got = ext.qsa_splitk_forward(query, key, value, selected, scale)

    def compare(name: str, other: torch.Tensor) -> dict:
        a, b = other.float(), ref.float()
        return {
            "max_abs_diff": float((a - b).abs().max()),
            "cosine_per_head": [
                float(torch.nn.functional.cosine_similarity(a[0, 0, h], b[0, 0, h], dim=0))
                for h in range(a.shape[2])
            ],
        }

    report["shipped_vs_torch"] = compare("shipped", want)
    report["splitk_vs_torch"] = compare("splitk", got)
    diff = (got.float() - want.float()).abs()
    report["max_abs_diff"] = float(diff.max())
    report["mean_abs_diff"] = float(diff.mean())
    report["matches"] = bool(torch.allclose(got.float(), want.float(), rtol=2e-2, atol=2e-2))
    # A cosine per head says whether the two agree in the direction that decides the argmax, which
    # is what a token comparison ultimately depends on.
    report["cosine_per_head"] = [
        float(torch.nn.functional.cosine_similarity(
            got[0, 0, h].float(), want[0, 0, h].float(), dim=0))
        for h in range(got.shape[2])
    ]

    # Time the shipped op and the prototype on the same one-slice workload.
    def timeit(fn, iters):
        for _ in range(10):
            fn()
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize(device)
        return 1000.0 * (time.perf_counter() - started) / iters

    report["shipped_ms_per_call"] = timeit(
        lambda: shipped.qwen4_exp_qsa_bf16_forward(query, key, value, selected, scale), args.iters)
    report["splitk_ms_per_call"] = timeit(
        lambda: ext.qsa_splitk_forward(query, key, value, selected, scale), args.iters)
    report["speedup"] = (report["shipped_ms_per_call"] / report["splitk_ms_per_call"]
                         if report["splitk_ms_per_call"] else None)

    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()