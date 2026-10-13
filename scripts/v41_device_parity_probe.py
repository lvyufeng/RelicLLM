"""Record the port's logits over the oracle's raw prompt, on a chosen expert path.

Both `--device` (the dense tree) and `--expert-device` (the routed experts) default to `None`, and
that pair of defaults is the host (correctness) path: the whole tree on CPU, `CheckpointRoutedExperts`
reading the experts out of the shards -- the path the loader calls the correctness path and the one
the oracle's offloaded `MoE.forward` mirrors. `CheckpointRoutedExperts` refuses an activation that is
not on the host, so the host expert path cannot be combined with a card-resident tree.
`scripts/v41_host_parity_probe.py` was this probe with both expert-path flags left unset, so it was
folded in here and the file removed; leave both flags off to run the path it pinned.

`--expert-device cuda:N` moves only the routed experts onto a card (`DeviceRoutedExperts`), which
consumes the packed fp4 rows a run stages to it; where the dense tree itself lives is `--device`'s
call, and the example below leaves `--device` unset, so it is a host tree with card-side experts.
Either path runs over the oracle's exact raw ids and dumps the per-step logit rows, so a run can be
diffed position by position against a reference dump -- `python scripts/record_reference_v41.py
--dump`, which writes `/tmp/ref_logits.pt` by default.

    PYTHONPATH=<relicllm checkout> python scripts/v41_device_parity_probe.py \
        --checkpoint <ckpt> --expert-device cuda:0 --out /tmp/port_dev.pt

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
    parser.add_argument("--out", default="/tmp/port_dev.pt")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--device", default=None, help="dense-tree device, e.g. cuda:0; None = host")
    parser.add_argument("--expert-device", default=None, help="staging device, e.g. cuda:0")
    parser.add_argument("--resident-engram", action="store_true")
    parser.add_argument(
        "--tokens",
        default="[19905,418,270,4031,2562,28,22471,45117,16]",
        help="JSON list of prompt token ids; the default is the oracle's raw prompt",
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from relicllm.models.deepseek_v4_1.config import load_config
    from relicllm.models.deepseek_v4_1.generate import generate
    from relicllm.models.deepseek_v4_1.loader import (
        V41Checkpoint,
        build_hasher,
        load_backbone,
    )

    # Default: the oracle's raw ids (scripts/record_reference_v41.py). `--tokens` overrides with some
    # other prompt, e.g. the chat-templated form the official `generate.py` actually feeds.
    ids = json.loads(args.tokens)

    def say(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    config = load_config(f"{args.checkpoint}/config.json").text
    _, hasher = build_hasher(config, args.checkpoint, tokenizer=tokenizer)
    checkpoint = V41Checkpoint(args.checkpoint)

    front = load_backbone(
        config,
        checkpoint,
        hasher=hasher,
        device=args.device,
        resident_engram=args.resident_engram,
        expert_device=args.expert_device,
        progress=say,
    )
    kinds = sorted({type(layer.ffn.routed).__name__ for layer in front.model.layers})
    say(
        f"routed experts: {', '.join(kinds)} on {len(front.model.layers)} layers; "
        f"device={args.device} expert_device={args.expert_device}"
    )

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
            "device": args.device,
            "expert_device": args.expert_device,
        },
        args.out,
    )
    say(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())