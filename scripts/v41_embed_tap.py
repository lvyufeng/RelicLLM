"""Record the port's step-0 forward stage by stage: the embedding, then every block's own stream.

The existing tap (`v41_layer_tap_probe.py`) reconstructs each layer's head projection through the
*last* block's `hc_pre`. That is the forward's arithmetic for exactly one layer -- the last -- so a
per-layer difference it reports for layer i is not layer i's block, and its layer-0 row is layer
0's stream pushed through a projection that belongs to layer 39. It answers "how far is this
layer's stream from the reference's" and not "is the port's block 0 correct", which is the question
once the embedding-level agreement is known to be near-perfect.

What this records needs no reconstruction and no head:

* `embed` -- the output of `self.embed(input_ids)`, before the `hc_mult` expansion. The one stage
  between the token ids and the model with no learned arithmetic in it, so it is the floor: if this
  disagrees, nothing above it can agree.
* `h[i]` -- what each `Block.forward` returns as its stream `[b, s, hc_mult, dim]`. Both sides
  return the same thing from the same place, so this compares like for like.
* `pre_mix[i]` -- the Hyper-Connections coefficients returned beside the stream, so the mixing
  itself is checkable rather than only its result.
* `attn[i]` / `ffn[i]` for the leading `--detail` blocks -- the block's two halves as the block
  used them, which attributes a block-level error to one of them.

Run on the box holding the released checkpoint, with the port's env:

    PYTHONPATH=<relic-core>:<relicllm> python scripts/v41_embed_tap.py \
        --checkpoint <ckpt> --out /tmp/port-embed-tap.pt

The reference side of the comparison is `scripts/ref_embed_tap.py`, which records the same keys
against the checkpoint's own `inference/model.py`.

Do not add `-I`: it implies `-E`, which drops `PYTHONPATH`, and the run then imports whichever tree
the environment's editable install points at rather than the one named above. A probe that reports
numbers from a revision nobody chose reads exactly like one that works.
"""
from __future__ import annotations

import argparse
import sys

import torch


def _quantized_holders(model):
    """Every object whose `_quantized` flag this forward actually reads.

    Not `model.modules()`: the host path's routed experts are a `CheckpointRoutedExperts`, which is
    deliberately not an `nn.Module` (its `forward` streams 384 experts off disk, so it wants no
    `Module` bookkeeping), and `MoE` holds it as a plain attribute -- `self.routed = ...`. A walk over
    `modules()` therefore clears attention, the indexer and the engram but leaves the flag the
    largest fp8 consumer in the model reads, and an A/B built on that walk measures a forward with
    three quarters of its quantization still on. So the walk descends one level into the plain
    attributes of every module as well.
    """
    seen: set[int] = set()
    for module in model.modules():
        for holder in (module, *vars(module).values()):
            if id(holder) in seen or not hasattr(holder, "_quantized"):
                continue
            seen.add(id(holder))
            yield holder


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="/tmp/port-embed-tap.pt")
    parser.add_argument("--detail", type=int, default=4,
                        help="how many leading blocks also record their attn/ffn sublayer outputs")
    parser.add_argument("--no-quantized", action="store_true",
                        help="clear every `_quantized` flag the forward reads before it runs, so the "
                             "run is the reference's arithmetic with the activation quantization "
                             "removed and nothing else changed; see `_quantized_holders` for why the "
                             "walk is not `model.modules()`")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from relicllm.models.deepseek_v4_1.config import load_config
    from relicllm.models.deepseek_v4_1.loader import V41Checkpoint, build_hasher, load_backbone

    ids = [19905, 418, 270, 4031, 2562, 28, 22471, 45117, 16]

    def say(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    config = load_config(f"{args.checkpoint}/config.json").text
    _, hasher = build_hasher(config, args.checkpoint, tokenizer=tokenizer)
    checkpoint = V41Checkpoint(args.checkpoint)

    # `expert_device=None` is the host path, the one the loader calls the correctness path.
    front = load_backbone(
        config, checkpoint, hasher=hasher, resident_engram=False, expert_device=None, progress=say,
    )
    model = front.model
    kinds = sorted({type(layer.ffn.routed).__name__ for layer in model.layers})
    say(f"routed experts: {', '.join(kinds)} on {len(model.layers)} layers")

    holders = list(_quantized_holders(model))
    if args.no_quantized:
        for holder in holders:
            holder._quantized = False
    say(f"objects carrying `_quantized`: {len(holders)} "
        f"({', '.join(sorted({type(h).__name__ for h in holders}))}) "
        f"(cleared: {args.no_quantized})")

    record: dict = {"embed": None, "h": {}, "pre_mix": {}, "attn": {}, "ffn": {}}
    handles = []

    def embed_hook(_module, _inputs, output):
        record["embed"] = output.detach().to(torch.float32).cpu().clone()

    handles.append(model.embed.register_forward_hook(embed_hook))

    def make_block(i):
        def hook(_module, _inputs, output):
            h, pre_mix = output
            record["h"][i] = h.detach().to(torch.float32).cpu().clone()
            record["pre_mix"][i] = pre_mix.detach().to(torch.float32).cpu().clone()
        return hook

    def make_sub(name, i):
        def hook(_module, _inputs, output):
            record[name][i] = output.detach().to(torch.float32).cpu().clone()
        return hook

    for i, block in enumerate(model.layers):
        handles.append(block.register_forward_hook(make_block(i)))
        if i < args.detail:
            # `attn` and `ffn` are each called once per block and each returns one tensor, so a
            # forward hook on the submodule is that sublayer's contribution and nothing else.
            handles.append(block.attn.register_forward_hook(make_sub("attn", i)))
            handles.append(block.ffn.register_forward_hook(make_sub("ffn", i)))

    with torch.inference_mode():
        _out, logits, _main = front(torch.tensor([ids]))
    for handle in handles:
        handle.remove()

    # The row `generate` would pick from, so the record says what this forward answered.
    record["step0_argmax"] = int(logits[0, -1].float().argmax())
    record["prompt_tokens"] = len(ids)
    say(f"captured embed + {len(record['h'])} of {len(model.layers)} blocks")
    say(f"step0 argmax {record['step0_argmax']}")

    torch.save(record, args.out)
    say(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
