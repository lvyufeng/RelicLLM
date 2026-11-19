"""Cross-host PD handoff proof; identical whole outputs are required, not projected speed.

Run on two leased TP8 hosts with the same checkpoint, Python sources and native extension.
The source freezes all rank-local state before capture and continues ordinary lookahead.
The destination loads those exact bytes and continues without a prefill forward.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def source_fingerprint(model_path):
    from relic_core.kernels.cuda_loader import load_cuda_kernel
    paths = [ROOT / "relicllm/models/qwen4_exp" / name for name in
             ("runtime.py", "model.py", "layers.py", "attention.py", "moe.py", "mtp.py",
              "verify_graph.py", "startup_graph.py", "pd_state.py")]
    module = load_cuda_kernel()
    if module is None:
        raise RuntimeError("the matching native extension must be loaded before this proof")
    files = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in paths}
    return {"code": files, "kernel_sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
            "config_sha256": hashlib.sha256((Path(model_path) / "config.json").read_bytes()).hexdigest()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("role", choices=("prefill", "decode"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--state-dir", required=True, type=Path)
    ap.add_argument("--context-tokens", type=int, default=8192)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--draft-tokens", type=int, default=3)
    ap.add_argument("--reference-tokens", type=Path)
    ap.add_argument("--output-jsonl", required=True, type=Path)
    args = ap.parse_args()
    if args.draft_tokens not in (3, 4) or args.context_tokens <= 0 or args.max_new_tokens <= 0:
        ap.error("the proof requires positive lengths and a qualified draft depth 3 or 4")
    import torch
    import torch.distributed as dist
    from transformers import AutoTokenizer
    from bench_qwen4_exp_baseline import _inputs
    from relicllm.models.qwen4_exp.pd_state import freeze_state, load_bundle, save_bundle, tensor_hashes
    from relicllm.models.qwen4_exp.runtime import init_distributed, load_model, generate_mtp_draft_lookahead

    ctx = init_distributed()
    if ctx.world_size != 8 or ctx.pp_size != 1 or ctx.device.type != "cuda":
        raise RuntimeError("this cross-host proof requires one separately authorized TP8 GPU host per role")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    inputs = _inputs(tokenizer, args.context_tokens)
    input_hash = hashlib.sha256(inputs.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    expected_ids = json.loads(args.reference_tokens.read_text()) if args.reference_tokens else None
    if expected_ids is not None and len(expected_ids) != args.max_new_tokens:
        raise ValueError("the complete reference must cover every requested output token")
    started = time.perf_counter()
    model, config = load_model(args.model, ctx, expert_resident=True, expert_fused_fp8=True,
                               graph_window=0, mtp=True)
    torch.cuda.synchronize(ctx.device)
    load_s = time.perf_counter() - started
    backbone_moe = getattr(model.layers[0].moe, "inner", model.layers[0].moe)
    if type(backbone_moe).__name__ != "NvFp4SlotExpertMoE":
        raise RuntimeError("the PD proof must retain the actual NVFP4 backbone expert bank")
    fingerprint = source_fingerprint(args.model)
    metadata = {**fingerprint, "input_sha256": input_hash, "draft_tokens": args.draft_tokens,
                "max_new_tokens": args.max_new_tokens, "chunk_size": args.chunk_size,
                "flags": {key: value for key, value in sorted(os.environ.items())
                          if key.startswith(("POCKETLLM_QWEN4_", "POCKETLLM_GRAPH_RETURN_MARGIN"))}}
    transfer = {}
    bundle = None
    path = args.state_dir / f"rank-{ctx.rank}.pd.zip"
    if args.role == "decode":
        bundle_started = time.perf_counter()
        bundle = load_bundle(path)
        if bundle.manifest.get("metadata") != metadata:
            raise ValueError("code, checkpoint, native binary or request does not match the source PD manifest")
        transfer = {"bundle_load_s": time.perf_counter() - bundle_started,
                    "state_file_bytes": path.stat().st_size}

    def export(state):
        nonlocal transfer
        args.state_dir.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError("a PD state artifact already exists; do not overwrite an uncertain handoff")
        export_started = time.perf_counter()
        frozen = freeze_state(
            cache=state["cache"], head_cache=state["head_cache"], past=state["past"],
            next_token=state["next_token"], first_draft=state["first_draft"],
            chain_stack=state["chain_stack"], first_conf=state["first_conf"], prefill_s=state["prefill_s"],
            tp_rank=ctx.rank, tp_world_size=ctx.world_size,
            native_max_context=int(config.text_config.max_position_embeddings), metadata=metadata,
        )
        freeze_s = time.perf_counter() - export_started
        written = save_bundle(frozen, path)
        transfer = {"freeze_s": freeze_s, "freeze_save_s": time.perf_counter() - export_started,
                    "state_payload_bytes": sum(t.numel() * t.element_size() for t in frozen.tensors.values()),
                    "state_file_bytes": written, "tensor_hashes": tensor_hashes(frozen),
                    "backbone_layers": len(frozen.manifest["backbone"]["layers"])}

    delivered = []
    output, stats = generate_mtp_draft_lookahead(
        model, ctx, inputs, max_new_tokens=args.max_new_tokens, chunk_size=args.chunk_size,
        draft_tokens=args.draft_tokens, verify_graph=True, on_token=delivered.append,
        eos_token_ids=(), prefilled_state=bundle, on_prefill=export if args.role == "prefill" else None,
    )
    local = {"rank": ctx.rank, "output_ids": output, "callback_ids": delivered,
             "transfer": transfer, "stats": stats, "load_s": load_s}
    ranks = [None] * ctx.world_size
    dist.all_gather_object(ranks, local)
    valid = (len(output) == args.max_new_tokens and ranks[0]["callback_ids"] == output
             and all(row["output_ids"] == output
                     and row["callback_ids"] == (output if row["rank"] == 0 else []) for row in ranks)
             and (expected_ids is None or expected_ids == output))
    if args.role == "decode":
        valid = (valid and bool(stats.get("pd_prefilled"))
                 and stats.get("pd_indexer_pool_needs_recompute") == 0
                 and stats.get("pd_indexer_pool_rebound") == 13)
    result = {"kind": "qwen4_exp_pd_handoff_result", "schema": "qwen4-exp-pd-handoff-v1",
              "role": args.role, "valid": valid, "requested_output_tokens": args.max_new_tokens,
              "input_tokens": int(inputs.shape[1]), "input_sha256": input_hash,
              "native_max_context": int(config.text_config.max_position_embeddings),
              "logical_total_tokens": int(inputs.shape[1]) + args.max_new_tokens,
              "reference_provided": expected_ids is not None, "output_ids": output,
              "callback_equal": delivered == output, "all_ranks_equal": all(
                  row["output_ids"] == output
                  and row["callback_ids"] == (output if row["rank"] == 0 else []) for row in ranks),
              "backbone_moe_type": type(backbone_moe).__name__,
              "head_moe_type": type(getattr(model.mtp.layer.moe, "inner", model.mtp.layer.moe)).__name__,
              "fingerprint": fingerprint, "raw_stats_rank0": stats, "per_rank": ranks,
              "measurement_scope": "full-state cross-host continuation; not streaming-overlap or production HTTP"}
    if ctx.rank == 0:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.output_jsonl.open("x") as stream:
            stream.write(json.dumps(result) + "\n")
        print(json.dumps(result), flush=True)
    dist.barrier()
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
