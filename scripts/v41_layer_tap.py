#!/usr/bin/env python3
"""Run the official v41 reference and apply the real head to each layer's collapsed stream.

When the port and the reference disagree, the output token says *that* they differ but not where.
This wraps every ``Block`` of DeepSeek's unmodified ``inference/model.py`` so that, after it returns,
the collapsed residual stream at the last position is captured and pushed through the *real* norm and
head. Layer ``i``'s entry is therefore the greedy answer the stack would give if it ended at layer
``i`` -- a per-layer localisation of a divergence, without a second oracle at each depth.

Same host-offload as ``scripts/v41_reference_linear_swap.py`` (identical MoE/Engram patches, identical
seeding), so a tap run and a swap run are comparable.

Usage, on the 4090 box where the converted TP checkpoint and the ``dsref`` env live::

    python scripts/v41_layer_tap.py --ckpt /path/to/DeepSeek-V4.1-Flash-TP8 \\
        --inference /path/to/DeepSeek-V4.1-Flash/inference --out /tmp/ref-tap.pt

Dumps ``{layer -> {"logits": [V] fp32, "h": [D] fp32, "norm": [D] fp32}}`` for step 0 only, plus the
step-0 argmax and the prompt length.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

#: The raw prompt, in step with ``scripts/record_reference_v41.py`` and the committed fixtures.
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


def install_tap(model):
    """Wrap every block so the collapsed stream at the last position is captured per layer.

    The head is applied at the end, after the whole prompt has run, so the tap costs one projection
    per layer and not one per layer per step. Only step 0 is captured.
    """
    taps = {}
    originals = [block.forward for block in model.layers]

    for layer_id, block in enumerate(model.layers):
        def make(i, orig):
            def forward(x, start_pos, pre_mix, image_mask, *a):
                h, pm = orig(x, start_pos, pre_mix, image_mask, *a)
                taps[i] = (h, pm)
                return h, pm
            return forward
        block.forward = make(layer_id, originals[layer_id])
    return taps, originals


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
    taps, originals = install_tap(model)

    torch.set_default_device("cuda")
    with torch.no_grad():
        cur = torch.tensor([TOKENS], dtype=torch.long, device=dev)
        out_ids, _, _ = model(cur, 0)
    nxt = int(out_ids.reshape(-1)[-1].item())
    if rank == 0:
        print("STEP0_ARGMAX", nxt, flush=True)

    # The head is a collective (`all_gather` when world_size > 1), so every rank has to run the
    # projections; only rank 0 keeps the record. Guarding the whole block on rank 0 hangs the other
    # ranks at the barrier while rank 0 waits in an all_gather nobody else entered.
    record = {}
    last = model.layers[-1]
    with torch.no_grad():
        for i in sorted(taps):
            h, pm = taps[i]
            collapsed = last.hc_pre(h, pm)  # [b, s, d]; hc_pre is a pure function of x and pre_mix
            normed = model.norm(collapsed)
            logits = model.head(normed)
            if rank == 0:
                record[i] = {
                    "h": collapsed[:, -1].float().cpu()[0].clone(),
                    "norm": normed[:, -1].float().cpu()[0].clone(),
                    "logits": logits[0].float().cpu().clone(),
                }
            del collapsed, normed, logits
    if rank == 0:
        torch.save({"layers": record, "step0_argmax": nxt, "prompt_tokens": len(TOKENS)}, OUT)
        print("TAP_DUMPED", OUT, "layers", len(record), flush=True)

    # restore so the harness is honest about not leaving patched forwards behind
    for block, orig in zip(model.layers, originals):
        block.forward = orig

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
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("/tmp/ref-tap.pt"))
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