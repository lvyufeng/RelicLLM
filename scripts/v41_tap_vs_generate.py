"""Settle whether the layer-tap reconstruction and `generate()`'s step-0 row are the same forward.

Two of our own port measurements disagree on the same prompt: `v41_layer_tap_probe.py` reconstructs a
top logit of 17.615 for token 455 from its captured layer-39 stream, while `v41_host_parity_probe.py`
-- and a second run on another box -- read 18.650 off the same step. Everything downstream (the
per-layer error profile, "the port is 2% off at layer 0") rests on the tap, so which of the two is the
port's real step-0 row has to be settled before either is used.

This runs all three in one process, on one loaded model, in one configuration:

* `bare_forward` -- what `front(torch.tensor([ids]))` returns, the call the tap script makes;
* `tap_reconstruction` -- the same forward's hook-captured block-39 `(h, pre_mix)` pushed through the
  last block's `hc_pre`, the model's norm and the head, which is what the tap's rows are;
* `generate_step0` -- the logits `generate()` hands `on_token` for the first new token, which is what
  the host-parity probe records.

If the first two differ, the tap rig is wrong and its per-layer profile means nothing. If the third
differs from both, step 0 is not one forward but two different ones and the comparison has to say
which. Run it on the box holding the released checkpoint, with the port's env:

    PYTHONPATH=<relicllm checkout> python scripts/v41_tap_vs_generate.py \
        --checkpoint <ckpt> --out /tmp/tap_vs_generate.pt

**Do not add `-I`.** `PYTHONPATH` is what selects the checkout: on a host with an editable install of
`relicllm`, the installed finder points at whichever tree was `pip install -e`'d, which need not be
the worktree under test. `PYTHONPATH` wins over it, and `-I` would drop `PYTHONPATH` and silently
measure the other tree -- a run that looks fine and reports numbers from a revision nobody chose.

The artifact carries the three rows, the `diff` computed from them, and a `config` block naming the
routed expert class that was built, the imported `relicllm` tree and its commit, the `relic_core`
those ops resolved out of, the `DEEPSEEK_*`/`POCKETLLM_*` switches in force, and the device. The
rows answer "is the tap the same forward" only for the configuration they came from, and this script
pins the host path (`expert_device=None`), so the block is what keeps a later reading from
attributing them to the device path -- or to a revision whose `relic_core` and knobs were not the
ones recorded beside them.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="/tmp/tap_vs_generate.pt")
    parser.add_argument("--resident-engram", action="store_true")
    args = parser.parse_args()

    import relic_core
    import relicllm
    from transformers import AutoTokenizer

    from relicllm.bench.metadata import cuda_metadata, env_snapshot, git_commit
    from relicllm.models.deepseek_v4_1.config import load_config
    from relicllm.models.deepseek_v4_1.generate import generate
    from relicllm.models.deepseek_v4_1.loader import V41Checkpoint, build_hasher, load_backbone

    # The oracle's raw ids, the same nine every parity probe uses.
    ids = [19905, 418, 270, 4031, 2562, 28, 22471, 45117, 16]

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
        resident_engram=args.resident_engram,
        expert_device=None,
        progress=say,
    )
    model = front.model
    kinds = sorted({type(layer.ffn.routed).__name__ for layer in model.layers})

    # Which path these three rows describe is part of the reading, not a detail of the run: the rows
    # only settle the tap-vs-generate question *for this configuration*. This script pins
    # `expert_device=None`, the host path, so a host-vs-device difference is not in scope here -- and
    # the artifact says so itself rather than relying on anyone's memory of the command line.
    tree = Path(relicllm.__file__).resolve().parent.parent
    core_tree = Path(relic_core.__file__).resolve().parent.parent
    run_config = {
        "expert_device": None,  # the argument `load_backbone` got; `routed` is what it built
        "routed": kinds,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "prompt_ids": ids,
        "resident_engram": bool(args.resident_engram),
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
        # And the switches in force, because several of them move the very rows these probes record:
        # the prefill sparse-attention knobs change step 0 and nothing else, so a step-0 reading
        # without them is not attributable to a revision.
        "knobs": {**env_snapshot("DEEPSEEK_"), **env_snapshot("POCKETLLM_")},
        "cuda": cuda_metadata(),
    }
    say(f"routed experts: {', '.join(kinds)} on {len(model.layers)} layers")
    say(f"config: {run_config}")

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
        out_ids, logits, _ = front(torch.tensor([ids]))
    for handle in handles:
        handle.remove()

    # `Backbone.forward` ends in `head(..., full_logits=False)`, which has already narrowed to the last
    # position, so `logits` is `[b, vocab]` -- indexing `[0, -1]` there reads the last *vocabulary*
    # entry, a 0-dim tensor whose argmax is always 0 and which `torch.topk(row, 8)` then rejects. Take
    # the row when the head narrowed, the position when it did not.
    bare_row = (logits[0] if logits.dim() == 2 else logits[0, -1]).float().cpu()
    last = model.layers[-1]
    collapsed = last.hc_pre(*taps[len(model.layers) - 1])
    recon_row = model.head(model.norm(collapsed), full_logits=True)[0, -1].float().cpu()

    # The same reconstruction for every layer, so this one run also carries the per-layer profile
    # `v41_layer_tap_probe.py` exists to produce.
    per_layer = {}
    for layer_id in sorted(taps):
        h, pre_mix = taps[layer_id]
        coll = last.hc_pre(h, pre_mix)
        nrm = model.norm(coll)
        per_layer[layer_id] = {
            "h": coll[0, -1].float().cpu().clone(),
            "norm": nrm[0, -1].float().cpu().clone(),
            "logits": model.head(nrm, full_logits=True)[0, -1].float().cpu().clone(),
        }

    front.reset_state(1)
    rows: list[torch.Tensor] = []

    def on_token(token_id: int, row: torch.Tensor) -> None:
        rows.append(row.detach().float().cpu())
        say(f"generate step {len(rows) - 1} pick={token_id}")

    result = generate(front, ids, max_new_tokens=1, temperature=0.0, on_token=on_token)
    gen_row = rows[0]

    say(f"bare_forward   argmax={int(bare_row.argmax())}")
    for name, row in (
        ("bare_forward", bare_row),
        ("tap_reconstruction", recon_row),
        ("generate_step0", gen_row),
    ):
        top = torch.topk(row, 8)
        say(f"{name}: top={top.indices.tolist()} vals={[round(float(v), 5) for v in top.values.tolist()]}")

    # The pairwise numbers, computed here rather than off to the side in a scratch cell: the answer to
    # "are these the same forward" is the artifact's own `diff`, not a figure quoted from elsewhere.
    bare_flat, recon_flat, gen_flat = (row.reshape(-1) for row in (bare_row, recon_row, gen_row))
    if not (bare_flat.numel() == recon_flat.numel() == gen_flat.numel()):
        raise SystemExit(f"rows have different lengths: {bare_row.shape} {recon_row.shape} {gen_row.shape}")
    diff = {
        "bare_vs_generate_ndiff": int((bare_flat != gen_flat).sum()),
        "elements": int(bare_flat.numel()),
        "recon_vs_bare_max_abs": float((recon_flat - bare_flat).abs().max()),
    }
    say(f"bare_forward vs generate_step0: {diff['bare_vs_generate_ndiff']}/{diff['elements']} elements differ")
    say(f"tap_reconstruction vs bare_forward: max |d| {diff['recon_vs_bare_max_abs']:.6g} nats")

    torch.save(
        {
            "bare_forward": bare_row,
            "tap_reconstruction": recon_row,
            "generate_step0": gen_row,
            "tokens": list(result.tokens),
            "prompt_tokens": len(ids),
            "diff": diff,
            "config": run_config,
        },
        args.out,
    )
    say(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())