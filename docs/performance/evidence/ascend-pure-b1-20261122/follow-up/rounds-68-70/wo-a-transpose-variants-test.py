"""Can the transpose be avoided while keeping the original arithmetic exactly?

The contiguous pre-transpose changed results, so the reduction order is not layout-invariant. Two more
candidates, both aimed at keeping the original matmul arithmetic while skipping the materialised write:
a transposed *view* handed to matmul, and the original einsum for reference.
"""
import sys
import time

import torch
import torch_npu  # noqa: F401

torch.npu.set_device(0)
device = torch.device('npu', 0)
groups, o_lora_rank, d = 2, 1024, 2048
bsz, seqlen = 1, 1


def reference(o, weight):
    return torch.einsum("bsgd,grd->bsgr", o, weight.view(groups, o_lora_rank, -1))


def transposed_view(o, weight):
    wo_a = weight.view(groups, o_lora_rank, -1)
    return torch.matmul(o, wo_a.transpose(1, 2))


def contiguous_copy(o, weight):
    wo_a_t = weight.view(groups, o_lora_rank, -1).transpose(1, 2).contiguous()
    return torch.einsum("bsgd,gdr->bsgr", o, wo_a_t)


variants = [('transposed view + matmul', transposed_view), ('contiguous copy + einsum', contiguous_copy)]
results = {name: 0 for name, _ in variants}
trials = 8
for trial in range(trials):
    generator = torch.Generator().manual_seed(500 + trial)
    o = torch.randn((bsz, seqlen, groups, d), generator=generator, dtype=torch.bfloat16).to(device)
    weight = torch.randn((groups * o_lora_rank, d), generator=generator, dtype=torch.bfloat16).to(device)
    want = reference(o, weight)
    for name, fn in variants:
        if torch.equal(want.cpu(), fn(o, weight).cpu()):
            results[name] += 1

print('over %d trials, bit-identical to the reference:' % trials)
for name, count in results.items():
    print('  %-28s %d/%d %s' % (name, count, trials, 'BIT-EXACT' if count == trials else 'not exact'))


def timed(fn, repeats=30):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    samples = []
    for _ in range(repeats):
        torch.npu.synchronize()
        start = time.perf_counter()
        fn()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - start) * 1e6)
    samples.sort()
    return samples[len(samples) // 2]


generator = torch.Generator().manual_seed(999)
o = torch.randn((bsz, seqlen, groups, d), generator=generator, dtype=torch.bfloat16).to(device)
weight = torch.randn((groups * o_lora_rank, d), generator=generator, dtype=torch.bfloat16).to(device)
print()
print('eager per-call cost (host bound, indicative only):')
for name, fn in [('reference einsum', reference)] + variants:
    print('  %-28s %8.1f us' % (name, timed(lambda fn=fn: fn(o, weight))))
