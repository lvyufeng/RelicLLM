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
"""
from __future__ import annotations

import argparse
import sys

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="/tmp/tap_vs_generate.pt")
    parser.add_argument("--resident-engram", action="store_true")
    args = parser.parse_args()

    from transformers import AutoTokenizer

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

    torch.save(
        {
            "bare_forward": bare_row,
            "tap_reconstruction": recon_row,
            "generate_step0": gen_row,
            "tokens": list(result.tokens),
            "prompt_tokens": len(ids),
        },
        args.out,
    )
    say(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())