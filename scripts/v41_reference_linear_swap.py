#!/usr/bin/env python3
"""Falsify the v41 port-vs-reference divergence by swapping one function in the official model.

`relicllm/models/deepseek_v4_1/modules.py` states the departure itself: the released checkpoint's
fp8/fp4 weights are "dequantized once, at load, and the arithmetic runs in bf16". The reference
instead keeps the weight quantized and quantizes the **activation** to fp8 before a block-scaled
GEMM (`inference/model.py::linear`). Those are two different arithmetics, and this script runs
DeepSeek's unmodified `model.py` with exactly one thing changed -- the `linear` function -- so the
two can be compared without a second variable.

Two variants:

``dequant-bf16``
    The port's arithmetic: dequantize the weight to bf16 and run a plain ``F.linear``. If the
    resulting step-0 argmax lands on the port's, the swap is the whole of the gap.

``actq-fp32``
    The reference's arithmetic, soft: quantize the activation through the checkpoint's own
    ``act_quant``, dequantize activation and weight by their block-32 E8M0 scales, matmul in fp32,
    cast to bf16. If this lands back on the official argmax, the activation quantization is the
    difference and the weights need not stay packed.

The model code is the checkpoint's own ``inference/model.py`` and ``inference/kernel.py``, imported
by path and never edited. The only other deviation is memory placement, and it is forced: a stock
forward needs ~62.5 GiB per rank that a 24 GiB card cannot hold, so the two sparse tables stay in
host RAM and their active slice streams to the card, differing from the released forward only by the
``.to(device)`` moves.

Usage, on the 4090 box where the converted TP checkpoint and the ``dsref`` env live::

    python scripts/v41_reference_linear_swap.py --variant dequant-bf16 \\
        --ckpt /path/to/DeepSeek-V4.1-Flash-TP8 \\
        --inference /path/to/DeepSeek-V4.1-Flash/inference

A human reads the printed top-8 against ``tests/fixtures/reference/v41.json`` (the official oracle).
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

#: The child run under ``torchrun``. A string because it must ``sys.path`` the checkpoint's own
#: ``inference`` dir before importing ``model`` -- a step that only makes sense on the recording host.
_CHILD_SOURCE = r'''
import json, os, sys

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import load_model

CKPT = os.environ["REF_CKPT"]
REF_DIR = os.environ["REF_INFERENCE_DIR"]
VARIANT = os.environ["REF_LINEAR_VARIANT"]
TOKENS = json.loads(os.environ["REF_TOKENS"])
TOP_K = int(os.environ.get("REF_TOP_K", "8"))

HOST_PATTERNS = (
    "ffn.experts.", ".engram.embed.", "mtp.",
    "vision.", "aligner.", "image_start", "image_end", "image_newline",
)

# The E2M1 codebook the checkpoint's fp4 packs into, in code order (sign is the high bit).
_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def is_host(name):
    return any(p in name for p in HOST_PATTERNS)


def dequant_fp8(w, s, bs=32):
    """[out, in] fp8 beside an [ceil(out/bs), ceil(in/bs)] E8M0 grid, expanded per element."""
    o, i = w.shape
    grid = s.float().repeat_interleave(bs, dim=0)[:o].repeat_interleave(bs, dim=1)[:, :i]
    return w.float() * grid


def dequant_fp4(w, s, bs=32):
    """[out, in//2] packed two-per-byte along K, with an [out, in//bs] E8M0 scale per 32 on K."""
    o, half = w.shape
    i = half * 2
    # `int8` holds the byte; the low nibble is the even element, the high nibble the odd one.
    byte = w.view(torch.uint8) if w.dtype == torch.float4_e2m1fn_x2 else w.to(torch.int8).view(torch.uint8)
    lo = (byte & 0x0F).to(torch.long)
    hi = ((byte >> 4) & 0x0F).to(torch.long)
    idx = torch.stack([lo, hi], dim=-1).reshape(o, i)
    table = torch.tensor(_E2M1, dtype=torch.float32, device=w.device)
    return table[idx] * s.float().repeat_interleave(bs, dim=1)[:, :i]


def make_linear(variant, ref_act_quant):
    def linear(x, weight, bias=None):
        assert bias is None
        if weight.dtype == torch.float4_e2m1fn_x2:
            w = dequant_fp4(weight, weight.scale)
        elif weight.dtype == torch.float8_e4m3fn:
            w = dequant_fp8(weight, weight.scale)
        else:
            return F.linear(x, weight)
        if variant == "dequant-bf16":
            return F.linear(x, w.to(torch.bfloat16))
        # actq-fp32: the reference's own activation quantization, dequantized back exactly as its
        # fp8_gemm does, then an fp32-accumulating matmul (which is what the kernel's FP32
        # accumulator produces), cast to bf16 as the kernel's bf16 out_dtype.
        k = x.size(-1)
        q, s = ref_act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
        xr = q.float() * s.float().repeat_interleave(32, dim=-1)[..., :k]
        return (xr @ w.T).to(torch.bfloat16)
    return linear


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
    import model as ref
    from model import MoE, ParallelEngramEmbedding, Transformer, ModelArgs
    from kernel import act_quant as ref_act_quant
    from transformers import AutoTokenizer

    ref.linear = make_linear(VARIANT, ref_act_quant)

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

    def patch_moe(module_cls):
        # Identical to the released MoE.forward except for the two .to() moves around each expert.
        def forward(self, x, image_mask=None):
            shape = x.size()
            x = x.view(-1, self.dim)
            weights, indices = self.gate(x, None if image_mask is None else image_mask.flatten())
            y = torch.zeros_like(x, dtype=torch.float32)
            counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
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

    patch_moe(MoE)
    patch_engram_embed(ParallelEngramEmbedding)

    torch.set_default_device("cuda")
    with torch.no_grad():
        cur = torch.tensor([TOKENS], dtype=torch.long, device=dev)
        out_ids, logits, _ = model(cur, 0)
    nxt = int(out_ids.reshape(-1)[-1].item())
    if rank == 0:
        step0 = logits[0].float().cpu()
        v, i = torch.topk(step0, TOP_K)
        print(f"SWAP_VARIANT {VARIANT}", flush=True)
        print("SWAP_ARGMAX", nxt, flush=True)
        print("SWAP_TOP8_IDS", json.dumps(i.tolist()), flush=True)
        print("SWAP_TOP8_VALUES", json.dumps([round(float(t), 4) for t in v]), flush=True)
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
    parser.add_argument("--variant", choices=("dequant-bf16", "actq-fp32"), default="actq-fp32")
    parser.add_argument("--world-size", type=int, default=0, help="0 = all visible cards")
    parser.add_argument("--python", default=sys.executable, help="interpreter that has tilelang")
    parser.add_argument("--tokens", default=json.dumps(DEFAULT_TOKENS),
                        help="raw prompt token ids, as a JSON list")
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args(argv)

    world = args.world_size or _visible_cards()
    env = {
        **dict(os.environ),
        "REF_CKPT": args.ckpt,
        "REF_INFERENCE_DIR": args.inference,
        "REF_LINEAR_VARIANT": args.variant,
        "REF_TOKENS": args.tokens,
        "REF_TOP_K": str(args.top_k),
        "OMP_NUM_THREADS": "8",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }

    print(f"swapping linear() -> {args.variant}: tp={world}", flush=True)
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