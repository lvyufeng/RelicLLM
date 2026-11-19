"""Can a layout-preserving batched BF16 sum reproduce serial decode collectives exactly?

Numerical diagnostic, not an NCCL tuning or throughput benchmark. No NCCL settings are changed.
Serial H-element all-reduces are the reference, including their BF16 intermediate rounding. Candidate
layouts permute K rows before one K*H-element collective. Exactness is checked on cancellation-heavy
and ordinary random values, eagerly and with captured serial/candidate CUDA graphs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from relicllm.models.qwen4_exp.runtime import init_distributed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=2560)
    ap.add_argument("--trials", type=int, default=8)
    args = ap.parse_args()
    ctx = init_distributed()
    rows, hidden = args.rows, args.hidden
    if ctx.tp_world_size != 8 or ctx.device.type != "cuda":
        raise RuntimeError("requires the leased eight-GPU tensor-parallel host")
    blocks = [block for block in [1, 16, 32, 64, 128, 160, 256, 320, 512, 640, 1280, hidden]
              if hidden % block == 0]
    torch.cuda.set_device(ctx.device)
    source = torch.zeros(rows, hidden, dtype=torch.bfloat16, device=ctx.device)

    def pattern(trial):
        gen = torch.Generator(device=ctx.device).manual_seed(9191 + trial * 17 + ctx.rank)
        x = torch.randn(rows, hidden, generator=gen, device=ctx.device)
        if trial % 4 == 1:
            # Adjacent huge signed operands and small residuals expose any re-association.
            x = x.sign() * (2.0 ** ((ctx.rank % 4) * 6 - 9))
        elif trial % 4 == 2:
            x = x * 0.01 + (8192 if ctx.rank % 2 == 0 else -8192)
            if ctx.rank in (1, 5):
                x *= 1.0009765625
        elif trial % 4 == 3:
            exponents = torch.randint(-12, 13, (rows, hidden), generator=gen, device=ctx.device)
            x = x * (2.0 ** exponents)
        return x.to(torch.bfloat16)

    def serial_into(out):
        out.copy_(source)
        for row in out:
            dist.all_reduce(row)

    def candidate_into(out, block):
        packed = source.view(rows, hidden // block, block).permute(1, 0, 2).contiguous().clone()
        # `block == hidden` makes the permuted view contiguous without a copy. Always own the
        # landing buffer: an in-place collective must not mutate the next reference's source.
        dist.all_reduce(packed)
        out.copy_(packed.permute(1, 0, 2).reshape(rows, hidden))

    records = []
    with torch.inference_mode():
        for block in blocks:
            reference = torch.empty_like(source)
            candidate = torch.empty_like(source)
            mismatch_eager = mismatch_captured = 0
            max_abs = 0.0
            source.copy_(pattern(0))
            serial_into(reference); candidate_into(candidate, block)
            torch.cuda.synchronize()
            stream = torch.cuda.Stream(device=ctx.device)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                serial_into(reference); candidate_into(candidate, block)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            serial_graph, candidate_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
            with torch.cuda.graph(serial_graph):
                serial_into(reference)
            with torch.cuda.graph(candidate_graph):
                candidate_into(candidate, block)
            for trial in range(args.trials):
                source.copy_(pattern(trial))
                serial_into(reference); candidate_into(candidate, block)
                unequal = (reference != candidate).sum().item()
                mismatch_eager += int(unequal)
                max_abs = max(max_abs, (reference.float() - candidate.float()).abs().max().item())
                serial_graph.replay(); candidate_graph.replay()
                mismatch_captured += int((reference != candidate).sum().item())
            totals = torch.tensor([mismatch_eager, mismatch_captured], device=ctx.device, dtype=torch.long)
            dist.all_reduce(totals, op=dist.ReduceOp.MAX)
            record = {"block_width": block, "rows": rows, "hidden": hidden, "trials": args.trials,
                      "different_elements_eager_max_rank": int(totals[0]),
                      "different_elements_captured_max_rank": int(totals[1]),
                      "max_abs_rank0": max_abs, "exact_in_this_test": bool((totals == 0).all().item())}
            records.append(record)
            if ctx.rank == 0:
                print(json.dumps({"kind": "bf16_collective_layout_diagnostic", **record}), flush=True)
            del serial_graph, candidate_graph
    if ctx.rank == 0:
        print(json.dumps({"kind": "bf16_collective_layout_summary", "exact_block_widths": [r["block_width"] for r in records if r["exact_in_this_test"]],
                          "not_a_general_exactness_proof": True, "not_a_throughput_measurement": True}), flush=True)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
