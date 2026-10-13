"""Record the port's host (correctness) expert path over the oracle's raw prompt.

The served-path fixture and the oracle disagree. The served path drives the routed experts from a
card (`DeviceRoutedExperts`, reached because the fixture passes `--tensor-parallel-size 4`), and that
class the port's own docs put 1.5-2.4% of the output scale from the host `CheckpointRoutedExperts`
path -- which the loader calls the correctness path and which is what the oracle's offloaded
`MoE.forward` mirrors. This probe runs that host path over the oracle's exact raw ids and dumps the
per-step logit rows, so the two can be compared position by position.

Not part of the runtime and not a test. Run it on the box holding the released checkpoint:

    PYTHONPATH=<relicllm checkout> python scripts/v41_host_parity_probe.py \
        --checkpoint /path/to/DeepSeek-V4.1-Flash --out /tmp/port_host.pt

Do not add `-I`: it implies `-E`, which drops `PYTHONPATH`, and the run then imports
whichever tree the environment's editable install points at rather than the one named above.
A probe that reports numbers from a revision nobody chose reads exactly like one that works.
"""
from __future__ import annotations

import argparse
import json
import sys

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="/tmp/port_host.pt")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--resident-engram", action="store_true")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from relicllm.models.deepseek_v4_1.config import load_config
    from relicllm.models.deepseek_v4_1.generate import generate
    from relicllm.models.deepseek_v4_1.loader import V41Checkpoint, build_hasher, load_backbone

    # The oracle's raw ids, and the prompt they tokenize from (scripts/record_reference_v41.py).
    ids = [19905, 418, 270, 4031, 2562, 28, 22471, 45117, 16]

    def say(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    config = load_config(f"{args.checkpoint}/config.json").text
    _, hasher = build_hasher(config, args.checkpoint, tokenizer=tokenizer)
    checkpoint = V41Checkpoint(args.checkpoint)

    # `expert_device=None` is the whole point: it is the host `CheckpointRoutedExperts` path, the
    # one the loader calls the correctness path and the one the oracle mirrors.
    front = load_backbone(
        config,
        checkpoint,
        hasher=hasher,
        resident_engram=args.resident_engram,
        expert_device=None,
        progress=say,
    )
    kinds = sorted({type(layer.ffn.routed).__name__ for layer in front.model.layers})
    say(f"routed experts: {', '.join(kinds)} on {len(front.model.layers)} layers")

    rows: list[torch.Tensor] = []

    def on_token(token_id: int, row: torch.Tensor) -> None:
        flat = row.detach().float().cpu()
        rows.append(flat)
        top = torch.topk(flat, 8)
        say(
            f"STEP {len(rows) - 1} pick={token_id} "
            f"top={top.indices.tolist()} vals={[round(v, 5) for v in top.values.tolist()]}"
        )

    result = generate(
        front,
        ids,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        eos_token_id=tokenizer.eos_token_id,
        on_token=on_token,
    )

    print("TOKENS", result.tokens, flush=True)
    print("STOPPED", result.stopped, flush=True)
    torch.save(
        {
            "tokens": list(result.tokens),
            "logits": torch.stack(rows) if rows else torch.empty(0),
            "prompt_tokens": len(ids),
            "path": kinds,
        },
        args.out,
    )
    say(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())