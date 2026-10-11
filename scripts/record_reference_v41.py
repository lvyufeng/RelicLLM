#!/usr/bin/env python3
"""Record the *official* DeepSeek-V4.1 reference answer for the v41 golden prompt.

This is not the served-path recorder. ``scripts/record_golden_fixture.py`` records what *this
repository* produces on a given card; this script records what **DeepSeek's own released code**
produces, so the repository has an oracle that is not itself a relicllm run. The two answer
different questions, and conflating them is how a port's own bias becomes its own standard.

What runs, and what does not
----------------------------

The model code is the checkpoint's own ``inference/model.py`` and ``inference/kernel.py``,
**unmodified** — imported by path, never edited. The one deviation is memory placement, and it is
forced rather than chosen: a stock forward needs ~62.5 GiB per rank and a single 24 GiB RTX 4090
cannot hold it, so the two sparse tables (routed experts, engram) are kept in host RAM and their
active slice is streamed to the card per step. That patch replaces ``MoE.forward`` and
``ParallelEngramEmbedding.forward`` with a version that differs from the released one **only by the
``tensor.to(device)`` moves** — the arithmetic, the clamps, the routing and the reduction order are
the checkpoint's. A reader can confirm that by diffing the two forward bodies against ``model.py``.

Why the raw prompt and not ``generate.py``
------------------------------------------

``inference/generate.py`` is the checkpoint's own entry point, but it renders the prompt through the
chat template (``encode_messages``), which for this prompt is 13 tokens — not the 9 the golden
fixture records. The fixture feeds the **raw** token ids, so the oracle for it must too. This script
seeds the reference ``Transformer`` with the same raw ids and reads the step-0 logits directly, which
is the quantity ``logits_check`` records.

Usage (on the 4090 box, where the converted TP8 checkpoint and the ``dsref`` env live)::

    python scripts/record_reference_v41.py \\
        --ckpt /path/to/DeepSeek-V4.1-Flash-TP8 \\
        --inference /path/to/DeepSeek-V4.1-Flash/inference \\
        --out tests/fixtures/reference/v41.json

A human reads the printed tokens and step-0 top-k before the record is committed, exactly as the
served-path recorder expects for its own answer.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import subprocess
import sys
import tempfile
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

#: The raw prompt. Kept in step with ``scripts/record_golden_fixture.py``'s ``DEFAULT_PROMPT`` and
#: the committed fixture's ``prompt``.
DEFAULT_PROMPT = "Reply with the single line: golden fixture."

#: ``DEFAULT_PROMPT`` under the checkpoint's tokenizer with ``add_special_tokens=False``. Pinned here
#: rather than re-tokenized so the oracle and the fixture share one input by construction.
DEFAULT_TOKENS = [19905, 418, 270, 4031, 2562, 28, 22471, 45117, 16]

#: The child run under ``torchrun`` on the recording host. A string because it must ``sys.path`` the
#: checkpoint's own ``inference`` dir before importing ``model`` — a step that only makes sense there.
_CHILD_SOURCE = r'''
import json, os, sys

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import load_model

CKPT = os.environ["REF_CKPT"]
REF_DIR = os.environ["REF_INFERENCE_DIR"]
TOKENS = json.loads(os.environ["REF_TOKENS"])
MAX_NEW = int(os.environ.get("REF_MAX_NEW", "16"))

# The sparse tables stay in host RAM: a stock forward needs far more than one 24 GiB card holds.
HOST_PATTERNS = (
    "ffn.experts.",      # routed experts: only the activated ones are ever streamed in
    ".engram.embed.",    # n-gram lookup table: only the hashed rows are gathered
    "mtp.",              # speculative draft blocks: a plain forward never touches them
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
    # Identical to the released gather except the table is read on the host and the rows are moved.
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

    # Build and load on the host, then promote only the dense weights to the card.
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

    torch.set_default_device("cuda")
    cur = torch.tensor([TOKENS], dtype=torch.long, device=dev)
    prev_pos = 0
    generated = []
    step0 = None
    all_logits = []
    for _ in range(MAX_NEW):
        out_ids, logits, _ = model(cur, prev_pos)
        all_logits.append(logits[0].detach().float().cpu())
        if step0 is None:
            step0 = logits[0].float().cpu()
        nxt = int(out_ids.reshape(-1)[-1].item())
        generated.append(nxt)
        if nxt == tokenizer.eos_token_id:
            break
        prev_pos += cur.shape[1]
        cur = torch.tensor([[nxt]], dtype=torch.long, device=dev)

    if rank == 0:
        torch.save(
            {"tokens": generated,
             "logits": torch.stack(all_logits),
             "prompt_tokens": len(TOKENS)},
            os.environ["REF_DUMP"],
        )
        print("REF_DUMPED", os.environ["REF_DUMP"], flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


main()
'''


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", required=True, help="the converted TP checkpoint directory")
    parser.add_argument("--inference", required=True,
                        help="the checkpoint's own inference/ dir (holds model.py, config.json)")
    parser.add_argument("--out", type=pathlib.Path,
                        default=REPO_ROOT / "tests" / "fixtures" / "reference" / "v41.json")
    parser.add_argument("--world-size", type=int, default=0, help="0 = all visible cards")
    parser.add_argument("--python", default=sys.executable, help="interpreter that has tilelang")
    parser.add_argument("--tokens", default=json.dumps(DEFAULT_TOKENS),
                        help="raw prompt token ids, as a JSON list")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--dump", type=pathlib.Path, default=pathlib.Path("/tmp/ref_logits.pt"))
    args = parser.parse_args(argv)

    tokens = json.loads(args.tokens)
    world = args.world_size or _visible_cards()
    env = {
        **dict(os.environ),
        "REF_CKPT": args.ckpt,
        "REF_INFERENCE_DIR": args.inference,
        "REF_TOKENS": json.dumps(tokens),
        "REF_MAX_NEW": str(args.max_tokens),
        "REF_DUMP": str(args.dump),
        "OMP_NUM_THREADS": "8",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }

    print(f"recording the official reference: tp={world}, {len(tokens)} raw tokens", flush=True)
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
    if proc.returncode != 0:
        print(f"the reference run exited {proc.returncode}", file=sys.stderr)
        return proc.returncode

    # Read the number back off the dump the child wrote rather than parsing its stdout.
    import torch

    dumped = torch.load(args.dump, map_location="cpu")
    values, ids = torch.topk(dumped["logits"][0].float(), args.top_k)
    payload: dict[str, Any] = {
        "entry": "v41",
        "source": "official DeepSeek-V4.1 reference (inference/model.py, unmodified)",
        "prompt": DEFAULT_PROMPT,
        "raw_token_ids": tokens,
        "tokens": [int(t) for t in dumped["tokens"]],
        "logits_check": {
            "step": 0,
            "prompt_tokens": int(dumped["prompt_tokens"]),
            "top_k": int(values.numel()),
            "token_ids": [int(i) for i in ids.tolist()],
            "values": [float(v) for v in values.tolist()],
        },
        "commit": _git_commit(),
        "taken_at": datetime.date.today().isoformat(),
        "notes": "the checkpoint's own model.py, imported unmodified and host-offloaded to fit a "
                 "24 GiB card; see scripts/record_reference_v41.py for the one deviation",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"  tokens = {payload['tokens']}")
    print(f"  step0  = {payload['logits_check']['token_ids']}")
    print(f"  values = {payload['logits_check']['values']}")
    return 0


def _visible_cards() -> int:
    import torch

    return torch.cuda.device_count() or 1


if __name__ == "__main__":
    raise SystemExit(main())