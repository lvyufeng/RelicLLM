#!/usr/bin/env python3
"""Record the reference's step-0 forward stage by stage, so it can be compared to the port's.

This is the reference side of ``scripts/v41_embed_tap.py``: same keys, same shapes, same positions,
produced by the checkpoint's own ``inference/model.py`` rather than by the port. It exists because
every comparison this project has made was between the port's reconstructed per-layer head
projection and the reference's, and that reconstruction borrows the *last* block's ``hc_pre`` -- so a
difference at layer ``i`` was never layer ``i``'s block. Recording the block's own returned stream
removes the reconstruction and with it the ambiguity.

What is recorded, and why each stage is the one to ask about:

* ``embed`` -- what ``self.embed`` returned, before the ``hc_mult`` expansion. Both sides return the
  same thing from the same place, and it is the one stage between the token ids and the model with no
  learned arithmetic in it, so it is the floor: if this disagrees, nothing above it can agree.
* ``h[i]`` -- what ``Block[i].forward`` returned as its stream, ``[b, s, hc_mult, dim]``. Neither the
  norm, the head, nor a neighbour's ``hc_pre`` touches it.
* ``pre_mix[i]`` -- the Hyper-Connections coefficients returned beside the stream, so the mixing
  itself is checkable rather than only its result.
* ``attn[i]`` / ``ffn[i]`` for the leading ``--detail`` blocks -- the block's two halves as the block
  used them, which attributes a block-level error to one of them.

Same host-offload as ``scripts/v41_layer_tap.py`` (identical MoE/Engram patches, identical seeding),
so a tap run and a swap run stay comparable.

Usage, on the 4090 box where the converted TP checkpoint and the ``dsref`` env live::

    python scripts/ref_embed_tap.py --ckpt /path/to/DeepSeek-V4.1-Flash-TP8 \\
        --inference /path/to/DeepSeek-V4.1-Flash/inference --out /tmp/ref-embed-tap.pt

Dumps ``{embed: [B,S,D], h/pre_mix/attn/ffn: {layer -> tensor}, step0_argmax, prompt_tokens}`` for
step 0 only.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile

#: The raw prompt, in step with ``scripts/v41_embed_tap.py`` and the committed fixtures.
DEFAULT_TOKENS = [19905, 418, 270, 4031, 2562, 28, 22471, 45117, 16]

_CHILD_SOURCE = r'''
import json, os, sys

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import load_model

CKPT = os.environ["REF_CKPT"]
REF_DIR = os.environ["REF_INFERENCE_DIR"]
TOKENS = json.loads(os.environ["REF_TOKENS"])
OUT = os.environ["REF_TAP_OUT"]
DETAIL = int(os.environ["REF_TAP_DETAIL"])

HOST_PATTERNS = (
    "ffn.experts.", ".engram.embed.", "mtp.",
    "vision.", "aligner.", "image_start", "image_end", "image_newline",
)


def is_host(name):
    return any(p in name for p in HOST_PATTERNS)


def patch_moe(module_cls):
    # Identical to the released MoE.forward except for the two .to() moves around each expert.
    def forward(self, x, image_mask=None):
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x, None if image_mask is None else image_mask.flatten())
        y = torch.zeros_like(x, dtype=torch.float32)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        dev = x.device
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            expert.to(dev)
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx], weights[idx, top, None])
            expert.to("cpu")
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(y)
        y += self.shared_experts(x)
        return y.type_as(x).view(shape)
    module_cls.forward = forward


def patch_engram_embed(module_cls):
    # Identical to the released gather except the table is read on the host and the rows moved.
    def forward(self, indices):
        idx = indices.cpu()
        mask = (idx < self.vocab_start_idx) | (idx >= self.vocab_end_idx)
        local = (idx - self.vocab_start_idx).masked_fill(mask, 0)
        values = F.embedding(local, self.weight)
        scales = F.embedding(local, self.scale)
        values = values.float().unflatten(-1, (-1, self.block_size)) * scales.float().unsqueeze(-1)
        values = values.flatten(-2).to(torch.bfloat16).to(indices.device)
        values = values.masked_fill(mask.to(indices.device).unsqueeze(-1), 0)
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(values)
        return values
    module_cls.forward = forward


def main():
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(33377335)

    sys.path.insert(0, REF_DIR)
    from model import MoE, ParallelEngramEmbedding, Transformer, ModelArgs
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(CKPT)
    with open(os.path.join(REF_DIR, "config.json")) as f:
        args = ModelArgs(**json.load(f))
    args.temperature = 0.0

    torch.set_default_device("cpu")
    model = Transformer(args, tokenizer)
    missing, unexpected = load_model(
        model, os.path.join(CKPT, f"model{rank}-mp{world_size}.safetensors"), strict=False
    )
    if rank == 0:
        print("LOAD_MISSING", len(missing), "LOAD_UNEXPECTED", len(unexpected), flush=True)

    dev = torch.device("cuda", local_rank)
    for name, p in model.named_parameters():
        if not is_host(name):
            p.data = p.data.to(dev)
    for name, b in model.named_buffers():
        b.data = b.data.to(dev)

    patch_moe(MoE)
    patch_engram_embed(ParallelEngramEmbedding)

    record = {"embed": None, "h": {}, "pre_mix": {}, "attn": {}, "ffn": {},
              "route": {}, "route_w": {}}
    handles = []

    # `model.embed` is a ParallelEmbedding: its forward all_reduces, so the hook sees the finished
    # embedding on every rank rather than one rank's shard.
    def embed_hook(_module, _inputs, output):
        record["embed"] = output.detach().float().cpu().clone()

    handles.append(model.embed.register_forward_hook(embed_hook))

    # `Block.forward` returns `(h, pre_mix)`; the hook reads the returned tuple, so it records what
    # the block produced rather than anything recomputed here.
    for i, block in enumerate(model.layers):
        def make_block(idx):
            def hook(_module, _inputs, output):
                h, pre_mix = output
                record["h"][idx] = h.detach().float().cpu().clone()
                record["pre_mix"][idx] = pre_mix.detach().float().cpu().clone()
            return hook
        handles.append(block.register_forward_hook(make_block(i)))
        for gate in (m for m in block.modules() if type(m).__name__ == "Gate"):
            # The router is a discrete decision: when two experts are close the top-k flips and the
            # block computes a different function rather than a perturbed one. Recorded per layer so
            # a block whose FFN moved further than its attention can be told apart from one that was
            # simply asked a different question.
            def make_gate(idx):
                def hook(_module, _inputs, output):
                    weights, indices = output
                    record["route"][idx] = indices.detach().to(torch.int32).cpu().clone()
                    record["route_w"][idx] = weights.detach().float().cpu().clone()
                return hook
            handles.append(gate.register_forward_hook(make_gate(i)))
        if i < DETAIL:
            def make_sub(name, idx):
                def hook(_module, _inputs, output):
                    record[name][idx] = output.detach().float().cpu().clone()
                return hook
            handles.append(block.attn.register_forward_hook(make_sub("attn", i)))
            handles.append(block.ffn.register_forward_hook(make_sub("ffn", i)))

    torch.set_default_device("cuda")
    with torch.no_grad():
        cur = torch.tensor([TOKENS], dtype=torch.long, device=dev)
        out_ids, logits, _ = model(cur, 0)
    for handle in handles:
        handle.remove()

    record["step0_argmax"] = int(logits[0].float().argmax())
    record["prompt_tokens"] = len(TOKENS)
    if rank == 0:
        print("STEP0_ARGMAX", record["step0_argmax"], flush=True)
        print(f"captured embed + {len(record['h'])} of {len(model.layers)} blocks", flush=True)
        torch.save(record, OUT)
        print("TAP_DUMPED", OUT, "layers", len(record["h"]), flush=True)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


main()
'''


def _visible_cards() -> int:
    import torch

    return torch.cuda.device_count() or 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", required=True, help="the converted TP checkpoint directory")
    parser.add_argument("--inference", required=True,
                        help="the checkpoint's own inference/ dir (holds model.py, config.json)")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("/tmp/ref-embed-tap.pt"))
    parser.add_argument("--detail", type=int, default=4,
                        help="how many leading blocks also record their attn/ffn sublayer outputs")
    parser.add_argument("--world-size", type=int, default=0, help="0 = all visible cards")
    parser.add_argument("--python", default=sys.executable, help="interpreter that has tilelang")
    parser.add_argument("--tokens", default=json.dumps(DEFAULT_TOKENS),
                        help="raw prompt token ids, as a JSON list")
    args = parser.parse_args(argv)

    world = args.world_size or _visible_cards()
    env = {
        **dict(os.environ),
        "REF_CKPT": args.ckpt,
        "REF_INFERENCE_DIR": args.inference,
        "REF_TOKENS": args.tokens,
        "REF_TAP_OUT": str(args.out),
        "REF_TAP_DETAIL": str(args.detail),
        "OMP_NUM_THREADS": "8",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }

    print(f"tapping the official reference: tp={world}", flush=True)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
        handle.write(_CHILD_SOURCE)
        child = handle.name
    try:
        proc = subprocess.run(
            [args.python, "-m", "torch.distributed.run", "--nproc_per_node", str(world), child],
            env=env,
        )
    finally:
        pathlib.Path(child).unlink(missing_ok=True)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())