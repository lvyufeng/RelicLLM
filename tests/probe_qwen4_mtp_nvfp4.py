"""Exact, streamed plain-vs-MTP measurements on an explicitly leased CUDA host.

Example (native total capacity, NOT a 262144-token input prompt):
  torchrun --nproc_per_node=8 tests/probe_qwen4_mtp_nvfp4.py \
    --model /workspace/qwen38-fp8 --context-tokens 262016 --max-new-tokens 128 \
    --draft-tokens 3 --arms plain,lag,lagq --reference /tmp/mtp-plain-262016.json

The lookahead runtime requests input + output + 2*K + 4 positions, clamps the arena to the native
maximum, and retires speculation for an exact one-row trunk tail before either head or trunk could
cross that maximum. This probe rejects input + requested output above native, reports both logical
length and allocated capacity, and accepts only the documented native-boundary tail as a captured
arm's eager suffix. It never shortens input/output or labels a nearly-256K prompt as a full 256K
prompt: 262144 input tokens plus 128 output tokens remains invalid for a native total maximum of
262144. Plain decode uses graph-window=32 by default. Lookahead uses graph-window=0 and its separate
verify_graph flag (lag/lagq). POCKETLLM_QWEN4_MTP_TAIL_GRAPH=1 opts into the setup-time one-row tail
recording. Its tail_graph_steps are counted separately from eager plain_tail_steps; both require
native-boundary position/reason metadata, and graph-tail steps additionally require the explicit
request flag and tail_graph_built=true. An eager suffix is never mislabeled as a captured tail.

Each arm prints a start record before prefill and a complete result immediately after it finishes,
including every output id, hashes, raw per-rank stats, numerator, denominator and stop reason. Rates
are measurements, never projections. --reference reuses an existing validated plain-output artifact,
or writes it after plain decode when absent. It saves repeated plain prefill, not MTP prefill. The
reference is invalidated by changed inputs, checkpoint metadata/configuration, runtime sources,
world size, math environment or baseline arguments. Shard metadata is hashed, not all weight bytes;
this is stated in the artifact. A rank exception is re-raised for torchrun to terminate its peers;
there is no catch-and-continue into mismatched collectives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
SCHEMA = "qwen4-exp-mtp-probe-v2"
ARMS = {"plain", "la", "lag", "lagq"}


def _hash_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _token_hash(ids: list[int]) -> str:
    return _hash_json([int(token) for token in ids])


def _first_mismatch(reference: list[int], output: list[int]) -> int | None:
    for position, (expected, actual) in enumerate(zip(reference, output)):
        if expected != actual:
            return position
    return min(len(reference), len(output)) if len(reference) != len(output) else None


def _completion(ids: list[int], requested: int, eos: tuple[int, ...]) -> dict:
    first_eos = next((i for i, token in enumerate(ids) if token in eos), None)
    natural_eos = first_eos is not None and first_eos == len(ids) - 1
    complete = 0 < len(ids) <= requested and (
        natural_eos or (len(ids) == requested and first_eos is None)
    )
    return {
        "stop_reason": "eos" if natural_eos else "budget" if len(ids) == requested else "incomplete",
        "complete": complete,
        "natural_eos": natural_eos,
        "first_eos_index": first_eos,
        "requested_output_tokens": requested,
        "emitted_tokens": len(ids),
    }


def _arena_plan(prompt: int, requested: int, native: int, arms: set[str], drafts: list[int]) -> dict:
    if min(prompt, requested, native) <= 0 or any(k <= 0 for k in drafts):
        raise ValueError("context, generation, native maximum and draft counts must be positive")
    if arms - ARMS or not arms:
        raise ValueError(f"--arms must name a nonempty subset of {sorted(ARMS)}")
    logical = prompt + requested
    if logical > native:
        raise ValueError(
            f"actual input {prompt} plus requested output {requested} is {logical}, "
            f"above native maximum {native}; largest permitted input is {native - requested}. "
            "Native total capacity is not the same as input-prompt length."
        )
    spans = {"plain": logical + 2}
    if arms - {"plain"}:
        if not drafts:
            raise ValueError("lookahead arms need at least one positive --draft-tokens value")
        spans.update({f"draft{k}": logical + 2 * k + 4 for k in drafts})
    clamped = {mode: min(span, native) for mode, span in spans.items()}
    maximum = max(clamped.values())
    return {"actual_input_tokens": prompt, "requested_output_tokens": requested,
            "requested_logical_total_tokens": logical,
            "native_max_position_embeddings": native, "arena_tokens_by_mode": clamped,
            "requested_arena_tokens_by_mode": spans,
            "arena_clamped_by_mode": {mode: span > native for mode, span in spans.items()},
            "largest_arena_tokens": maximum, "native_limit_respected": True,
            "boundary_policy": "retire speculation before native boundary; exact one-row trunk tail",
            "input_prompt_is_at_least_256k": prompt >= 262144,
            "logical_total_reaches_256k": logical >= 262144,
            "full_arena_reaches_256k": maximum >= 262144}


def _captured_execution(stats: dict, plan: dict, draft: int, emitted: int,
                        *, tail_graph_requested: bool = False) -> dict:
    """Accept graph rounds plus proved native-boundary row-one graph/eager suffixes only."""
    graph = stats.get("graph_rounds", 0)
    tail = stats.get("plain_tail_steps", 0)
    tail_graph = stats.get("tail_graph_steps", 0)
    tail_total = tail + tail_graph
    trunk = stats.get("trunk_steps", 0)
    eager = stats.get("mtp_eager_steps", trunk - graph - tail_total)
    counts = (graph, tail, tail_graph, trunk, eager)
    integral = all(type(value) in (int, float) and math.isfinite(value)
                   and value >= 0 and value == int(value) for value in counts)
    arena = plan["arena_tokens_by_mode"][f"draft{draft}"]
    native = plan["native_max_position_embeddings"]
    prompt = plan["actual_input_tokens"]
    delivered_total = prompt + emitted
    layout = (stats.get("arena_tokens") == arena
              and stats.get("native_max_context") == native
              and stats.get("logical_total_tokens") == delivered_total
              and delivered_total <= native)
    start = stats.get("plain_tail_start")
    reason = stats.get("plain_tail_reason", "")
    tail_valid = tail_total == 0
    if tail_total > 0:
        start_valid = (type(start) in (int, float) and math.isfinite(start) and start == int(start)
                       and prompt <= start < delivered_total - 1)
        tail_valid = (plan["arena_clamped_by_mode"][f"draft{draft}"] and arena == native
                      and reason == "native_boundary" and start_valid
                      and start + 2 * (draft + 1) > native
                      and tail_total == delivered_total - 1 - start)
    built = stats.get("tail_graph_built", False)
    graph_tail_valid = ((built is not True or tail_graph_requested)
                        and (tail_graph == 0 or (tail_graph_requested and built is True)))
    if tail_graph_requested:
        graph_tail_valid = (graph_tail_valid and type(built) is bool
                            and "tail_graph_steps" in stats and "tail_graph_built" in stats)
    valid = (integral and layout and graph > 0 and eager == 0
             and graph + tail_total == trunk and tail_valid and graph_tail_valid)
    return {"valid": bool(valid), "graph_rounds": graph, "native_eager_tail_steps": tail,
            "tail_graph_steps": tail_graph, "native_tail_steps_total": tail_total,
            "tail_graph_requested": tail_graph_requested, "tail_graph_built": built,
            "tail_graph_metadata_valid": bool(graph_tail_valid),
            "native_tail_fully_captured": bool(tail_total > 0 and tail_graph == tail_total),
            "nonboundary_mtp_eager_steps": eager, "trunk_steps": trunk,
            "tail_start": start, "tail_reason": reason,
            "tail_is_native_boundary": bool(tail_total > 0 and tail_valid),
            "runtime_native_layout_matches_plan": layout}


def _arm_order(arms: set[str], drafts: list[int]) -> list[tuple[str, int, bool]]:
    # NVFP4 conversion of the head is one-way. Never run an FP8-head arm afterwards.
    return [(mode, k, mode != "la") for mode in ("la", "lag", "lagq") if mode in arms
            for k in drafts]


def _source_hashes() -> dict[str, str]:
    directory = ROOT / "relicllm/models/qwen4_exp"
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.glob("*.py"))}


def _checkpoint_identity(directory: Path) -> dict:
    names = ("config.json", "model.safetensors.index.json", "tokenizer.json",
             "tokenizer_config.json", "special_tokens_map.json", "generation_config.json")
    small = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
             for name in names if (directory / name).is_file()}
    shards = [{"name": path.name, "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
              for path in sorted(directory.glob("*.safetensors"))]
    return {"model_directory": str(directory.resolve()), "configuration_sha256": small,
            "checkpoint_shard_metadata_sha256": _hash_json(shards), "checkpoint_shards": shards,
            "weight_identity_scope": "configuration bytes plus shard names/sizes/mtime_ns; not weight-byte hashes"}


def _math_environment() -> dict[str, str]:
    prefixes = ("POCKETLLM_", "RELICLLM_", "NCCL_", "CUBLAS_", "TORCH_", "CUDA_")
    return {key: value for key, value in sorted(os.environ.items()) if key.startswith(prefixes)}


def _kernel_identity() -> dict:
    """Fingerprint the loaded native binary and its Python arithmetic wrapper, not a guessed build."""
    from relic_core.kernels import ops
    from relic_core.kernels.cuda_loader import load_cuda_kernel

    native = load_cuda_kernel()
    paths = {"ops": Path(ops.__file__).resolve()}
    if native is None or not getattr(native, "__file__", None):
        raise RuntimeError("cannot fingerprint the loaded CUDA kernel extension")
    paths["native"] = Path(native.__file__).resolve()
    identity = {}
    for name, path in paths.items():
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        identity[name] = {"path": str(path), "sha256": digest.hexdigest()}
    return identity


def _reference_material(*, checkpoint: dict, source_hashes: dict, input_ids: list[int],
                        native: int, budget: int, eos: tuple[int, ...], args,
                        world_size: int, torch_version: str, cuda_version: str | None,
                        kernel_identity: dict | None = None) -> dict:
    # verify_graph.py is not executed by plain generate. Changing only the captured speculative
    # implementation must not force another eight-minute plain prefill.
    baseline_sources = {name: value for name, value in source_hashes.items() if name != "verify_graph.py"}
    return {"schema": SCHEMA, "checkpoint": checkpoint, "baseline_sources": baseline_sources,
            "input_sha256": _token_hash(input_ids), "actual_input_tokens": len(input_ids),
            "requested_output_tokens": args.max_new_tokens, "chunk_size": args.chunk_size,
            "plain_graph_window": args.graph_window, "native_max_position_embeddings": native,
            "indexer_budget": budget, "eos_token_ids": list(eos), "world_size": world_size,
            "torch_version": torch_version, "cuda_version": cuda_version,
            "kernel_identity": kernel_identity, "math_environment": _math_environment()}


def _validate_reference(document: dict, material: dict, eos: tuple[int, ...]) -> dict:
    if document.get("schema") != SCHEMA or document.get("kind") != "plain_reference":
        raise ValueError("reference is not a supported plain-output artifact")
    if document.get("fingerprint") != _hash_json(material) or document.get("material") != material:
        raise ValueError("plain reference fingerprint differs from this run; explicitly rebuild it")
    record = document.get("record", {})
    ids = record.get("output_ids")
    if not isinstance(ids, list) or any(type(token) is not int or token < 0 for token in ids):
        raise ValueError("reference has no valid complete output token list")
    if record.get("output_sha256") != _token_hash(ids) or record.get("arm") != "plain":
        raise ValueError("reference output hash or plain-arm identity is invalid")
    completion = _completion(ids, material["requested_output_tokens"], eos)
    if not completion["complete"] or not record.get("valid") or not record.get("required_execution_mode_ran"):
        raise ValueError("reference is incomplete or failed its execution/validation checks")
    return record


def _write_reference(path: Path, material: dict, record: dict) -> None:
    # A same-directory atomic replace never leaves a partial reference after an interrupted write.
    document = {"kind": "plain_reference", "schema": SCHEMA, "fingerprint": _hash_json(material),
                "material": material, "record": record}
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(document, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _rate_evidence(output: list[int], stats: dict) -> dict:
    duration = float(stats.get("decode_s", 0.0))
    # Preserve the runtime's raw numerator even on older trees whose lookahead incorrectly counts
    # the prefill-produced first token. The separately named normalized rate never counts it.
    numerator = stats.get("decode_tokens", stats.get("generated_tokens", len(output)))
    normalized = max(0, len(output) - 1)
    setup = float(stats.get("graph_setup_s", stats.get("graph_build_s", 0.0)))
    # New runtime stats make the scope explicit: graph_setup_s already includes the alignment
    # barrier, and decode_wall_s already includes graph_setup_s. Never add either a second time.
    wall = float(stats.get("decode_wall_s", duration + setup if "graph_build_s" in stats else duration))
    cleanup = stats.get("request_cleanup_s")
    inclusive = duration + float(cleanup) if cleanup is not None else None
    return {"runtime_decode_tps": stats.get("decode_tps"), "runtime_rate_numerator_tokens": numerator,
            "runtime_rate_denominator_s": duration, "emitted_minus_prefill_token_numerator": normalized,
            "emitted_minus_prefill_token_tps": normalized / duration if duration > 0 else None,
            "reported_decode_tps": stats.get("decode_tps"), "graph_setup_s": setup,
            "graph_build_s": stats.get("graph_build_s"),
            "graph_alignment_s": stats.get("graph_alignment_s"),
            "decode_wall_s_including_graph_setup": wall,
            "runtime_decode_wall_tps": stats.get("decode_wall_tps"),
            "emitted_minus_prefill_token_wall_tps": normalized / wall if wall > 0 else None,
            "runtime_decode_s_excluding_setup": stats.get("decode_s_excluding_setup"),
            "request_cleanup_s": cleanup,
            "deferred_graph_cleanup_s": stats.get("deferred_graph_cleanup_s"),
            "runtime_decode_s_including_cleanup": inclusive,
            "emitted_minus_prefill_token_tps_including_cleanup":
                normalized / inclusive if inclusive is not None and inclusive > 0 else None,
            "normalized_rate_scope": "(returned output length - 1) / runtime decode_s; excludes prefill "
                                     "and post-delivery resource cleanup; see the separately reported "
                                     "cleanup-inclusive rate and decode_wall_s for setup scope; "
                                     "alignment is included in graph_setup_s, not added separately",
            "rate_rank": 0, "measured_not_projected": True}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--context-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--graph-window", type=int, default=32, help="plain decode graph window only")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--draft-tokens", default="3")
    parser.add_argument("--warmup-s", type=float, default=3.0)
    parser.add_argument("--label", default="mtp")
    parser.add_argument("--arms", default="plain,lag", help="plain,la,lag,lagq; plain is the reference")
    parser.add_argument("--reference", type=Path, help="reuse existing plain artifact, or create after plain decode")
    parser.add_argument("--refresh-reference", action="store_true", help="explicitly replace the plain artifact")
    parser.add_argument("--reference-only", action="store_true", help="measure/load plain and stop")
    parser.add_argument("--output-jsonl", type=Path, help="new rank-0 evidence file (exclusive creation)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    arms = {mode.strip() for mode in args.arms.split(",") if mode.strip()}
    drafts = list(dict.fromkeys(int(value) for value in args.draft_tokens.split(",") if value.strip()))
    tail_graph_requested = os.environ.get("POCKETLLM_QWEN4_MTP_TAIL_GRAPH", "0") == "1"
    if args.reference_only:
        arms = {"plain"}
    if args.refresh_reference and args.reference is None:
        raise ValueError("--refresh-reference requires --reference")
    if args.chunk_size <= 0 or args.graph_window < 0:
        raise ValueError("chunk size must be positive and graph window nonnegative")
    if os.environ.get("POCKETLLM_QWEN4_MTP_NVFP4", "0") != "0":
        raise ValueError("start with POCKETLLM_QWEN4_MTP_NVFP4 unset/0; lagq enables the one-way swap")

    import torch
    import torch.distributed as dist
    from transformers import AutoTokenizer
    from bench_qwen4_exp_baseline import _cuda_memory, _inputs
    from relicllm.models.qwen4_exp.runtime import (
        _warmup_device, generate, generate_mtp_draft_lookahead, init_distributed, load_model,
    )

    ctx = init_distributed()
    if ctx.pp_size != 1 or ctx.device.type != "cuda":
        raise RuntimeError("requires pure TP on an explicitly leased CUDA host")
    evidence_file = args.output_jsonl.open("x") if ctx.rank == 0 and args.output_jsonl else None

    def emit(record):
        if ctx.rank == 0:
            line = json.dumps(record, sort_keys=True, allow_nan=False)
            print(line, flush=True)
            if evidence_file is not None:
                evidence_file.write(line + "\n")
                evidence_file.flush()

    def collect(value):
        if dist.is_initialized():
            gathered = [None] * ctx.world_size
            dist.all_gather_object(gathered, value)
            return gathered
        return [value]

    def result(mode: str, draft: int, output, stats: dict, elapsed: float,
               reference: list[int] | None, callbacks: list[int], emitted_at: list[float]) -> dict:
        output = [int(token) for token in output]
        ranks = collect({"rank": ctx.rank, "stats": stats, "output_ids": output, "cuda": _cuda_memory()})
        mismatch = _first_mismatch(reference, output) if reference is not None else None
        ranks_agree = all(rank["output_ids"] == output for rank in ranks)
        completion = _completion(output, args.max_new_tokens, eos)
        captured = mode in {"lag", "lagq"}
        capture_evidence = []
        if captured:
            capture_evidence = [_captured_execution(rank["stats"], plan, draft, len(output),
                                                    tail_graph_requested=tail_graph_requested)
                                for rank in ranks]
            executed = all(evidence["valid"] for evidence in capture_evidence)
        elif mode == "plain" and args.graph_window > 0 and len(output) > 1:
            executed = all(rank["stats"].get("replay_count", 0) > 0 for rank in ranks)
        else:
            executed = True
        streaming_agrees = callbacks == output if ctx.rank == 0 else True
        intervals = [later - earlier for earlier, later in zip(emitted_at, emitted_at[1:])]
        record = {"kind": "qwen4_exp_mtp_arm_result", "schema": SCHEMA, "label": args.label,
                  "arm": mode if mode == "plain" else f"{mode}_draft{draft}", "draft_tokens": draft,
                  **plan, **completion, **_rate_evidence(output, stats),
                  "output_ids": output, "output_sha256": _token_hash(output), "input_sha256": input_hash,
                  "matches_plain": mismatch is None, "first_mismatch": mismatch,
                  "reference_output_length": len(reference) if reference is not None else len(output),
                  "all_ranks_same_output": ranks_agree, "required_execution_mode_ran": executed,
                  "captured_execution_evidence_per_rank": capture_evidence,
                  "delivered_logical_total_tokens": len(ids) + len(output),
                  "verify_graph_requested": captured, "plain_graph_window": args.graph_window,
                  "tail_graph_requested": captured and tail_graph_requested,
                  "lookahead_graph_window": 0, "indexer_budget": 2048,
                  "effective_indexer_budgets": sorted(budgets), "wall_s": elapsed,
                  "raw_stats_rank0": ranks[0]["stats"], "ranks": ranks,
                  "rank0_callback_ids": callbacks if ctx.rank == 0 else None,
                  "rank0_callbacks_match_returned_output": streaming_agrees,
                  "rank0_inter_emission_s": intervals,
                  "rank0_streaming_tps": len(intervals) / sum(intervals) if intervals and sum(intervals) > 0 else None,
                  "reference_fingerprint": _hash_json(material), "source_hashes": sources}
        record["valid"] = completion["complete"] and mismatch is None and ranks_agree and executed and streaming_agrees
        # Rank 0 owns callbacks; make its validity the decision every rank takes before the next arm.
        if dist.is_initialized():
            status = [record["valid"] if ctx.rank == 0 else None]
            dist.broadcast_object_list(status, src=0)
            record["valid"] = bool(status[0])
        emit(record)
        return record

    try:
        started = time.perf_counter()
        model, config = load_model(args.model, ctx, expert_resident=True, expert_fused_fp8=True,
                                   graph_window=args.graph_window, mtp=True)
        load_s = time.perf_counter() - started
        if not model.has_mtp:
            raise RuntimeError("model was built without its MTP head")
        native = int(config.text_config.max_position_embeddings)
        budgets = {layer.attn.indexer.budget for layer in model.layers
                   if layer is not None and hasattr(layer.attn, "indexer")}
        if config.text_config.indexer_budget != 2048 or budgets != {2048}:
            raise RuntimeError(f"native effective indexer budget must remain 2048, got {budgets}")
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
        inputs = _inputs(tokenizer, args.context_tokens)
        ids = [int(token) for token in inputs[0].tolist()]
        if inputs.ndim != 2 or inputs.shape[0] != 1 or len(ids) != args.context_tokens:
            raise RuntimeError("constructed input does not have the exact requested token length")
        plan = _arena_plan(len(ids), args.max_new_tokens, native, arms, drafts)
        eos = tuple(int(value) for value in config.text_config.eos_token_id)
        input_hash = _token_hash(ids)
        sources = _source_hashes()
        material = _reference_material(checkpoint=_checkpoint_identity(Path(args.model)), source_hashes=sources,
                                       input_ids=ids, native=native, budget=2048, eos=eos, args=args,
                                       world_size=ctx.world_size, torch_version=torch.__version__,
                                       cuda_version=torch.version.cuda, kernel_identity=_kernel_identity())
        emit({"kind": "qwen4_exp_mtp_start", "schema": SCHEMA, "label": args.label, **plan,
              "requested_input_tokens": args.context_tokens, "input_sha256": input_hash,
              "load_s": load_s, "reference_material": material, "arms": sorted(arms),
              "draft_tokens": drafts, "plain_graph_window": args.graph_window,
              "lookahead_graph_window": 0})
        _warmup_device(ctx=ctx, dtype=model.dtype, seconds=args.warmup_s)
        reference_record = None
        if args.reference and args.reference.exists() and not args.refresh_reference:
            reference_record = _validate_reference(json.loads(args.reference.read_text()), material, eos)
            emit({"kind": "qwen4_exp_mtp_reference_reused", "schema": SCHEMA,
                  "reference_path": str(args.reference.resolve()), "fingerprint": _hash_json(material),
                  "output_ids": reference_record["output_ids"],
                  "output_sha256": reference_record["output_sha256"],
                  "emitted_tokens": len(reference_record["output_ids"]),
                  "no_new_plain_measurement": True})
        records = []
        if reference_record is None:
            emit({"kind": "qwen4_exp_mtp_arm_start", "arm": "plain", **plan})
            cache = model.make_cache(1, plan["arena_tokens_by_mode"]["plain"])
            callbacks, emitted_at = [], []

            def on_plain(token):
                callbacks.append(int(token))
                emitted_at.append(time.perf_counter())

            started = time.perf_counter()
            output, stats = generate(model=model, ctx=ctx, input_ids=inputs, chunk_size=args.chunk_size,
                                     max_new_tokens=args.max_new_tokens, graph_window=args.graph_window,
                                     cache=cache, eos_token_ids=eos, on_token=on_plain if ctx.rank == 0 else None)
            torch.cuda.synchronize(ctx.device)
            reference_record = result("plain", 0, output, stats, time.perf_counter() - started,
                                      None, callbacks, emitted_at)
            records.append(reference_record)
            del cache
            torch.cuda.empty_cache()
            if not reference_record["valid"]:
                emit({"kind": "qwen4_exp_mtp_summary", "valid": False, "failed_arm": "plain", "exit_code": 2})
                return 2
            if args.reference and ctx.rank == 0:
                _write_reference(args.reference, material, reference_record)
            if dist.is_initialized():
                dist.barrier()
        reference = reference_record["output_ids"]
        for mode, draft, captured in _arm_order(arms, drafts):
            if mode == "lagq":
                os.environ["POCKETLLM_QWEN4_MTP_NVFP4"] = "1"
            emit({"kind": "qwen4_exp_mtp_arm_start", "arm": f"{mode}_draft{draft}",
                  "verify_graph_requested": captured, "head_nvfp4_requested": mode == "lagq",
                  "tail_graph_requested": captured and tail_graph_requested, **plan})
            callbacks, emitted_at = [], []

            def on_draft(token):
                callbacks.append(int(token))
                emitted_at.append(time.perf_counter())

            started = time.perf_counter()
            output, stats = generate_mtp_draft_lookahead(
                model=model, ctx=ctx, input_ids=inputs, max_new_tokens=args.max_new_tokens,
                chunk_size=args.chunk_size, eos_token_ids=eos, graph_window=0, draft_tokens=draft,
                verify_graph=captured, on_token=on_draft if ctx.rank == 0 else None,
            )
            torch.cuda.synchronize(ctx.device)
            record = result(mode, draft, output, stats, time.perf_counter() - started,
                            reference, callbacks, emitted_at)
            records.append(record)
            torch.cuda.empty_cache()
            if not record["valid"]:
                emit({"kind": "qwen4_exp_mtp_summary", "valid": False,
                      "failed_arm": record["arm"], "exit_code": 2})
                return 2
        emit({"kind": "qwen4_exp_mtp_summary", "schema": SCHEMA, "valid": True, **plan,
              "completed_arms": [record["arm"] for record in records],
              "plain_reference_reused": not any(record["arm"] == "plain" for record in records),
              "reference_fingerprint": _hash_json(material), "exit_code": 0})
        return 0
    except Exception as exc:
        # No CUDA synchronization here: an exception can leave peers waiting in a collective.
        print(json.dumps({"kind": "qwen4_exp_mtp_rank_error", "rank": ctx.rank,
                          "error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr, flush=True)
        raise
    finally:
        if evidence_file is not None:
            evidence_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
