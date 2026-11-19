"""Price a recorded MTP round from a `POCKETLLM_QWEN4_ROUND_TRACE` chrome trace.

Every kernel a graph replay launches carries the correlation id of its `cudaGraphLaunch`, so the
kernels of each replay can be attributed to the `round.verify` / `round.draft` range that issued
it. Per range the report gives the device span of a replay (first kernel start to last kernel end),
the busy time inside it (the sum of kernel durations), and the kernels that take the time. The span
minus the busy time is what the replay spends between kernels.

Usage: python tests/analyze_round_trace.py trace.json [--top 25]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict


def _family(name: str) -> str:
    """Collapse template noise so kernels of one kind sum together."""
    name = re.sub(r"<.*", "", name)
    name = re.sub(r"\(.*", "", name)
    return name.strip()[:90]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()
    with open(args.trace) as fh:
        data = json.load(fh)
    events = data["traceEvents"] if isinstance(data, dict) else data

    ranges = [e for e in events if e.get("cat") == "user_annotation"
              and str(e.get("name", "")).startswith("round.")]
    launches = [e for e in events if e.get("cat") == "cuda_runtime"
                and e.get("name") in ("cudaGraphLaunch", "cudaGraphLaunch_v10000")]
    kernels = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]

    label_of = {}
    for launch in launches:
        ts = launch["ts"]
        for r in ranges:
            if r["ts"] <= ts <= r["ts"] + r.get("dur", 0):
                label_of[launch["args"]["correlation"]] = r["name"]
                break

    per_launch = defaultdict(list)
    for k in kernels:
        corr = k.get("args", {}).get("correlation")
        if corr in label_of:
            per_launch[corr].append(k)

    report = {}
    for label in sorted({v for v in label_of.values()}):
        corrs = [c for c, v in label_of.items() if v == label and per_launch.get(c)]
        if not corrs:
            continue
        spans, busy, counts = [], [], []
        by_name = defaultdict(float)
        nccl = 0.0
        for c in corrs:
            ks = per_launch[c]
            start = min(k["ts"] for k in ks)
            end = max(k["ts"] + k.get("dur", 0) for k in ks)
            spans.append(end - start)
            busy.append(sum(k.get("dur", 0) for k in ks))
            counts.append(len(ks))
            for k in ks:
                fam = _family(k["name"])
                by_name[fam] += k.get("dur", 0)
                if "nccl" in k["name"].lower():
                    nccl += k.get("dur", 0)
        n = len(corrs)
        top = sorted(by_name.items(), key=lambda kv: -kv[1])[: args.top]
        report[label] = {
            "replays": n,
            "span_ms": round(sum(spans) / n / 1000, 3),
            "busy_ms": round(sum(busy) / n / 1000, 3),
            "nccl_ms": round(nccl / n / 1000, 3),
            "kernels_per_replay": round(sum(counts) / n, 1),
            "top": [(name, round(us / n / 1000, 3)) for name, us in top],
        }
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
