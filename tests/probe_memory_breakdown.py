"""What a single rank is actually holding on an 8x24 GiB box.

The aggregate is not the question. Eight 24 GiB cards hold 192 GiB and the FP8 weights are far less
than that, so "not enough memory" has to mean something specific about one rank's 24 GiB. This
prints the per-rank post-build picture and attributes the weights by module category, so the split
between expert bank, dense half and everything else is measured rather than assumed.
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GIB = 1 << 30


def gib(n):
    return round(n / GIB, 3)


def tensor_bytes(t):
    try:
        return t.nbytes
    except AttributeError:
        return t.numel() * t.element_size()


def main():
    out = {"kind": "memory_breakdown"}
    from relicllm.models.qwen4_exp.runtime import init_distributed, load_model

    ctx = init_distributed()
    torch.cuda.synchronize()
    out["before_build"] = {"free_gib": gib(torch.cuda.mem_get_info()[0])}

    model, config = load_model(
        "<checkpoint>", ctx, expert_resident=True, expert_fused_fp8=True,
        graph_window=16, mtp=False,
    )
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    out["after_build"] = {
        "allocated_gib": gib(allocated),
        "reserved_gib": gib(torch.cuda.memory_reserved()),
        "free_gib": gib(torch.cuda.mem_get_info()[0]),
        "total_gib": gib(torch.cuda.mem_get_info()[1]),
    }
    if ctx.rank != 0:
        return

    buckets = {}
    seen = set()
    for name, tensor in list(model.state_dict().items()):
        key = id(tensor)
        if key in seen:
            continue
        seen.add(key)
        lower = name.lower()
        if "expert" in lower:
            bucket = "experts"
        elif "ple" in lower or "embed" in lower or "ngram" in lower:
            bucket = "embedding_host_expected"
        elif "indexer" in lower or "attn" in lower or "qsa" in lower:
            bucket = "attention"
        elif "delta" in lower or "gdn" in lower or "ssm" in lower:
            bucket = "gated_delta"
        elif "hc" in lower or "hyper" in lower:
            bucket = "hyper_connection"
        else:
            bucket = "other"
        buckets[bucket] = buckets.get(bucket, 0) + tensor_bytes(tensor)
    out["weights_by_category_gib"] = {k: gib(v) for k, v in sorted(buckets.items())}
    out["weights_counted_gib"] = gib(sum(buckets.values()))
    out["note"] = "allocated - counted = non-parameter device memory (arenas, workspace, graph pool)"
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
