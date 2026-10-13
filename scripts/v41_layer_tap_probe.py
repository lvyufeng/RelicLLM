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

The artifact also carries a `config` block -- the routed expert class that was built, the imported
`relicllm` tree and its commit, the `relic_core` those ops resolved out of, the
`DEEPSEEK_*`/`POCKETLLM_*` switches in force, the device, the prompt ids -- because a step-0 reading
is only evidence about the path it ran on, and two of those switches (the prefill
sparse-attention pair) move step 0 and nothing else. This script pins `expert_device=None`, the host
path the oracle's offloaded `MoE.forward` mirrors; the block is what makes that checkable from the
file alone.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="/tmp/port-tap.pt")
    args = parser.parse_args()

    import relic_core
    import relicllm
    from transformers import AutoTokenizer

    from relicllm.bench.metadata import cuda_metadata, env_snapshot, git_commit
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

    # A step-0 number is only evidence about *a path*, so the artifact has to say which one ran: the
    # routed class the model actually built (not the argument anyone remembers passing), the
    # `relicllm` tree that got imported -- see the `-I` note above for why that is not necessarily
    # the one named on the command line -- its commit, and the device. Without this, a host-path run
    # reads exactly like a device-path one, which is how a label drifted once already.
    tree = Path(relicllm.__file__).resolve().parent.parent
    core_tree = Path(relic_core.__file__).resolve().parent.parent
    run_config = {
        "expert_device": None,  # the argument `load_backbone` got; `routed` is what it built
        "routed": kinds,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "prompt_ids": ids,
        # Pinned at the `load_backbone` call above rather than taken from a flag, and written down
        # for the same reason the sibling records it: the Engram table is the other big host-resident
        # structure, so which of the two placements ran belongs in the reading.
        "resident_engram": False,
        "source": str(tree),
        "revision": git_commit(str(tree)),
        # The other half of "which code produced these rows". `relicllm` resolves its ops out of
        # whichever `relic_core` won the import, and an editable install pointing at a scratch
        # checkout answers a different kernel than the tree on the command line -- the same trap the
        # `-I` note above describes, one layer down.
        "relic_core": {
            "source": str(core_tree),
            "revision": git_commit(str(core_tree)),
        },
        # And the switches in force, because several of them move the very row this probe records:
        # the prefill sparse-attention knobs change step 0 and nothing else, so a step-0 reading
        # without them is not attributable to a revision.
        "knobs": {**env_snapshot("DEEPSEEK_"), **env_snapshot("POCKETLLM_")},
        # Where the tree itself ran, which is not what `cuda_metadata` answers: that is the box's
        # card inventory, and a host-path run on a four-card box reports four cards it never used.
        "device": str(model.device),
        "cuda": cuda_metadata(),
    }
    say(f"config: {run_config}")

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
            "config": run_config,
        },
        args.out,
    )
    say(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())