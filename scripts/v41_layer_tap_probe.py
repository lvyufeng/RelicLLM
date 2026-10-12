"""Record the port's per-layer step-0 tap, the mirror of the oracle's `ref-tap.py`.

Both step-0 logit rows exist and differ by a uniform ~2-3, while the decode steps agree -- so the
divergence is confined to how step 0's collapsed stream reaches the head, and localising it needs a
layer-by-layer trace, not another end-to-end token comparison. The reference side already has one:
`ref-tap.py` wraps each `Block`, keeps the `(h, pre_mix)` it returns, and pushes the collapsed last
position through the *real* norm and head. This probe does the same against the port, with identical
semantics, so `{layer -> h, norm, logits}` can be diffed layer by layer against `/tmp/ref-tap.pt`.

Run on the box holding the released checkpoint, with the port's env:

    PYTHONPATH=<relicllm checkout> python scripts/v41_layer_tap_probe.py \
        --checkpoint <ckpt> --out /tmp/port-tap.pt

Do not add `-I`: it implies `-E`, which drops `PYTHONPATH`, and the run then imports
whichever tree the environment's editable install points at rather than the one named above.
A probe that reports numbers from a revision nobody chose reads exactly like one that works.

`step0_argmax` is the argmax of the logit row the forward produced, not `front(...)`'s second return
value: `Backbone.forward` ends in `sample(logits, self.temperature)` with `temperature` defaulting to
1.0, so a bare forward's `output_ids` is one unseeded Gumbel draw over that row. Reading it as an
argmax is what produced a reported "1999" that no configuration reproduces -- the row beside it
argmaxes to 1. `generate()` pins the temperature to 0.0 and is the only path whose tokens are greedy
picks. The draw is recorded beside the argmax as `step0_sampled`, so nothing is silently dropped.
"""
from __future__ import annotations

import argparse
import sys

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="/tmp/port-tap.pt")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from relicllm.models.deepseek_v4_1.config import load_config
    from relicllm.models.deepseek_v4_1.loader import V41Checkpoint, build_hasher, load_backbone

    # The oracle's raw ids (scripts/record_reference_v41.py), same as every other parity probe.
    ids = [19905, 418, 270, 4031, 2562, 28, 22471, 45117, 16]

    def say(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    config = load_config(f"{args.checkpoint}/config.json").text
    _, hasher = build_hasher(config, args.checkpoint, tokenizer=tokenizer)
    checkpoint = V41Checkpoint(args.checkpoint)

    # `expert_device=None` is the host path, the one the loader calls the correctness path and the one
    # the oracle's offloaded `MoE.forward` mirrors.
    front = load_backbone(
        config,
        checkpoint,
        hasher=hasher,
        resident_engram=False,
        expert_device=None,
        progress=say,
    )
    model = front.model
    kinds = sorted({type(layer.ffn.routed).__name__ for layer in model.layers})
    say(f"routed experts: {', '.join(kinds)} on {len(model.layers)} layers")

    # Each Block returns `(h, ffn_pre)`; keep both, exactly as ref-tap.py does.
    taps: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    handles = []

    def make(i):
        def hook(_module, _inputs, output):
            h, pre_mix = output
            taps[i] = (h.detach(), pre_mix.detach())
        return hook

    for layer_id, block in enumerate(model.layers):
        handles.append(block.register_forward_hook(make(layer_id)))

    with torch.inference_mode():
        output_ids, logits, _ = front(torch.tensor([ids]))
    for handle in handles:
        handle.remove()

    # `Head` already narrowed to the last position (`full_logits=False`), so `logits` is `[b, vocab]`
    # and its argmax is the greedy step-0 answer. `output_ids` is the sampled draw beside it.
    step0_argmax = int(logits.argmax(dim=-1).reshape(-1)[-1].item())
    step0_sampled = int(output_ids.reshape(-1)[-1].item())
    say(f"STEP0_ARGMAX {step0_argmax}")
    say(f"STEP0_SAMPLED {step0_sampled} (unseeded temperature-1.0 draw; not the greedy pick)")
    say(f"captured {len(taps)} of {len(model.layers)} layers")

    # The reference's tap applies the *last* block's `hc_pre`, then the model's own norm and head, to
    # every layer's captured stream -- `hc_pre` is a pure function of x and pre_mix. Same here.
    last = model.layers[-1]
    record = {}
    for i in sorted(taps):
        h, pre_mix = taps[i]
        collapsed = last.hc_pre(h, pre_mix)  # [b, s, d]
        normed = model.norm(collapsed)
        row = model.head(normed, full_logits=True)[0, -1].float().cpu().clone()
        record[i] = {
            "h": collapsed[0, -1].float().cpu().clone(),
            "norm": normed[0, -1].float().cpu().clone(),
            "logits": row,
        }
        del collapsed, normed

    torch.save(
        {
            "layers": record,
            "step0_argmax": step0_argmax,
            "step0_sampled": step0_sampled,
            "prompt_tokens": len(ids),
        },
        args.out,
    )
    say(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())