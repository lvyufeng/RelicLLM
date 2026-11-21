"""Compare two completed actual generations without claiming official parity."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", required=True)
    parser.add_argument("--second", required=True)
    args = parser.parse_args()
    first, second = Path(args.first).resolve(), Path(args.second).resolve()
    payloads = []
    for folder in (first, second):
        summary = json.loads((folder / "summary.json").read_bytes())
        if summary["returncode"] != 0 or not summary["cleanup"]["complete"]:
            raise RuntimeError("cannot compare an unsuccessful or still-live inference")
        payloads.append(torch.load(folder / "generated-logits.pt", map_location="cpu", weights_only=True))
    one, two = payloads
    a, b = one["logits"].float(), two["logits"].float()
    if a.shape != b.shape or not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise RuntimeError("incomplete or non-finite logits")
    unchanged_sources = []
    conditions = json.loads((second / "preconditions.json").read_bytes())
    for name, expected in conditions["source_hashes"].items():
        actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError("a final-runtime source changed after the recorded inference: " + name)
        unchanged_sources.append(name)
    mismatches = int((a.contiguous().view(torch.int32) != b.contiguous().view(torch.int32)).sum())
    result = {
        "first_attempt": str(first), "second_attempt": str(second),
        "first_token_ids": [int(v) for v in one["tokens"]],
        "second_token_ids": [int(v) for v in two["tokens"]],
        "token_ids_equal": one["tokens"] == two["tokens"],
        "all_logits_finite": True, "logits_shape": list(a.shape),
        "float32_logit_bit_mismatches": mismatches,
        "max_absolute_logit_difference": float((a - b).abs().max()),
        "second_inference_source_hashes_still_match": True,
        "checked_source_count": len(unchanged_sources),
        "official_reference_parity_claimed": False,
        "repeat_scope": "two short untemplated original-checkpoint requests only",
    }
    with (second / "repeat-comparison.json").open("x") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
