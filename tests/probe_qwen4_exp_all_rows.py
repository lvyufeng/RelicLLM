"""Locate the first row-dependent arithmetic difference between a K-row verify forward and K
independent single-row decode steps from the same committed prefix.

Why this probe exists. The earlier `probe_qwen4_exp_first_row.py` compared a single reference row
against a batch, and only at a 1024-token context whose indexer stays under its budget; every
comparison it printed was row 0 of a repeated input, so it could not answer which row of a wider
verify diverges, nor whether the per-position recurrent/PLE state a rewind copies is the state
single-row decode would have produced. This probe feeds distinct teacher tokens, compares every
layer for every row, and checks the exact rewind contract:
`VerifyGraph.rewind` copies a recorded window into `conv_state` and a recorded recurrent state, so
that recorded pair must equal the state single-row decode has after the same position.

The teacher prefix comes from a validated plain-artifact output: the base context is built by
`bench_qwen4_exp_baseline._inputs`, then `--deliver-prefix` produced tokens are fed one at a time.
Single-row feeding is deliberate: a batched prefill of generated tokens has different arithmetic
from decode and would not reproduce the reference cache. The probe then runs both paths from one
snapshot and reports the first difference. It is a diagnostic: no TPS, no rate, no runtime change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
SCHEMA = "qwen4-exp-all-rows-v1"
FLAG_PREFIXES = ("POCKETLLM_QWEN4_", "POCKETLLM_GRAPH_RETURN_MARGIN")


# ------------------------------------------------------------------------------------------------
# Pure helpers: no torch and no model import at module scope, so the unit tests run without a card.
# ------------------------------------------------------------------------------------------------


def token_hash(ids: list[int]) -> str:
    """The same token-list hash the MTP probe writes as `input_sha256`/`output_sha256`."""
    return hashlib.sha256(
        json.dumps([int(token) for token in ids], sort_keys=True,
                   separators=(",", ":")).encode()
    ).hexdigest()


def flag_snapshot(environ: dict | None = None) -> dict:
    """The experiment flags in force, so a result is attributable to one configuration."""
    source = os.environ if environ is None else environ
    return {key: source[key] for key in sorted(source)
            if key.startswith(FLAG_PREFIXES)}


def _token_list(value: Any) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError("a reference needs a nonempty output token list")
    for token in value:
        if type(token) is not int or token < 0:
            raise ValueError("reference output ids must be non-negative integers")
    return [int(token) for token in value]


def _records(path: Path) -> list:
    """Read a JSON document or a JSON-lines file and return every record it holds."""
    text = path.read_text()
    try:
        return [json.loads(text)]
    except json.JSONDecodeError:
        records = []
        for line in text.splitlines():
            if line.strip():
                records.append(json.loads(line))
        return records


def load_teacher_reference(path: str | Path) -> dict:
    """Extract validated plain-arm output ids and the base-input hash from a reference artifact.

    Accepts the probe's `plain_reference` artifact, a raw probe JSONL (the plain arm result plus
    its start record), or any JSON holding `output_ids`. The start record's `input_sha256` is the
    hash of the base input tokens, which is what lets a caller prove the teacher prefix was
    produced from the same context this probe rebuilds.
    """
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"teacher reference does not exist: {source}")
    ids = None
    input_sha256 = None
    input_tokens = None
    source_kind = None
    output_sha256 = None
    # A raw probe JSONL holds every arm, so a generic `output_ids` match must not win over the
    # arm that actually defines plain greedy output. Collection is therefore two-tiered: specific
    # artifacts first, and only then a generic document, which is all a bare artifact offers.
    ids_preference = 0
    for record in _records(source):
        if not isinstance(record, dict):
            if ids is None or ids_preference < 1:
                ids, source_kind = _token_list(record), "bare_token_list"
                ids_preference = 1
            continue
        if record.get("kind") == "plain_reference":
            inner = record.get("record")
            if not isinstance(inner, dict):
                raise ValueError("plain_reference artifact has no record")
            material = record.get("material") or {}
            if ids is None or ids_preference < 4:
                ids, source_kind = _token_list(inner.get("output_ids")), "plain_reference"
                output_sha256 = inner.get("output_sha256")
                ids_preference = 4
            input_sha256 = material.get("input_sha256", input_sha256)
            input_tokens = material.get("actual_input_tokens", input_tokens)
        elif record.get("kind") == "qwen4_exp_mtp_start":
            input_sha256 = record.get("input_sha256", input_sha256)
            input_tokens = record.get("actual_input_tokens", input_tokens)
        elif record.get("kind") == "qwen4_exp_mtp_arm_result" and record.get("arm") == "plain":
            recorded_input = record.get("input_sha256")
            if recorded_input:
                input_sha256 = recorded_input
            if ids is None or ids_preference < 3:
                ids, source_kind = _token_list(record.get("output_ids")), "probe_plain_arm"
                output_sha256 = record.get("output_sha256")
                ids_preference = 3
        elif ids is None or ids_preference < 1:
            if "output_ids" in record:
                ids, source_kind = _token_list(record["output_ids"]), "generic_output_ids"
                output_sha256 = record.get("output_sha256", output_sha256)
                ids_preference = 1
    if ids is None:
        raise ValueError(f"no plain output token list found in {source}")
    if output_sha256 is not None and output_sha256 != token_hash(ids):
        raise ValueError("reference output hash does not match its token list")
    return {"path": str(source), "source_kind": source_kind, "ids": ids,
            "input_sha256": input_sha256, "input_tokens": input_tokens}


def select_window(ids: list[int], deliver_prefix: int, rows: int) -> dict:
    """Split reference output ids into the consumed prefix and the K-row verify window.

    Plain decode has consumed `output[0:deliver_prefix]` before the window, so window token `j` is
    fed at one position past the prefix and must predict the next reference token.
    """
    if deliver_prefix < 0:
        raise ValueError("deliver prefix cannot be negative")
    if rows < 1:
        raise ValueError("rows must be positive")
    needed = deliver_prefix + rows
    if len(ids) < needed:
        raise ValueError(
            f"reference has {len(ids)} output tokens; prefix {deliver_prefix} plus {rows} rows "
            f"needs {needed}"
        )
    predictions = ids[deliver_prefix + 1:needed + 1]
    return {"prefix_ids": [int(token) for token in ids[:deliver_prefix]],
            "feed_ids": [int(token) for token in ids[deliver_prefix:needed]],
            "teacher_predictions": [int(token) for token in predictions]}


def plan_arena(context_tokens: int, deliver_prefix: int, rows: int,
               native_max: int = 0, slack: int = 16) -> dict:
    """The request arena: prefix positions, the K-row span, and room for the rewind slack.

    The span needs slack because every QSA layer advances its own KV length by the rows it consumed;
    the logical total stays `context + deliver_prefix + rows` and is what a native bound applies to.
    """
    if context_tokens <= 0 or deliver_prefix < 0 or rows < 1 or slack < 0:
        raise ValueError("context must be positive; prefix/rows/slack cannot be negative")
    requested = context_tokens + deliver_prefix + rows + 2 * rows + slack
    logical_total = context_tokens + deliver_prefix + rows
    arena = requested
    if native_max and arena > native_max:
        if logical_total > native_max:
            raise ValueError(
                f"logical total {logical_total} exceeds the native maximum {native_max}"
            )
        arena = int(native_max)
    return {"arena_tokens": int(arena), "requested_arena_tokens": int(requested),
            "arena_clamped": int(arena) != requested, "logical_total_tokens": int(logical_total),
            "native_max_context": int(native_max)}


def validate_args(args) -> None:
    if not args.model:
        raise ValueError("a checkpoint path is required")
    if not args.teacher_reference:
        raise ValueError("a teacher reference artifact is required")
    if not 2 <= args.rows <= 8:
        raise ValueError("rows must be in [2, 8]: the row-invariant helpers cover 2..8 rows")
    if args.context_tokens <= 0:
        raise ValueError("context tokens must be positive")
    if args.deliver_prefix < 0:
        raise ValueError("deliver prefix cannot be negative")
    if args.chunk_size <= 0:
        raise ValueError("chunk size must be positive")
    if args.arena_slack < 0:
        raise ValueError("arena slack cannot be negative")


def compare_tensor(reference, candidate) -> dict:
    """Exact comparison; a leading-dimension difference is compared by flattened values.

    A rewind copies a recorded window into the live buffer, so a recorded leading singleton can
    legitimately meet a buffer without one. Those are compared value-by-value after flattening;
    a same-shape pair is compared with `torch.equal`, never a tolerance.
    """
    import torch

    result = {"equal": False, "reference_shape": list(reference.shape),
              "candidate_shape": list(candidate.shape),
              "reference_dtype": str(reference.dtype), "candidate_dtype": str(candidate.dtype)}
    if reference.shape == candidate.shape:
        left, right = reference, candidate
    elif reference.numel() == candidate.numel():
        left, right = reference.reshape(-1), candidate.reshape(-1)
    else:
        raise ValueError(
            "cannot compare tensors with different element counts: "
            f"{list(reference.shape)} ({reference.numel()}) and "
            f"{list(candidate.shape)} ({candidate.numel()})"
        )
    result["equal"] = bool(torch.equal(left, right))
    if not result["equal"]:
        result["differing_elements"] = int((left != right).sum().item())
        if left.is_floating_point() and right.is_floating_point():
            result["max_abs"] = float((right.float() - left.float()).abs().max().item())
    return result


def layer_row_differences(reference_layers: list, candidate_layers: list) -> list:
    """Every differing `(layer, row)`, in layer-then-row order -- not only row zero.

    Inputs are `(layer_index, tensor[rows, width])` pairs recorded from the batch forward. The first
    element is therefore the first row-dependent divergence in layer order, and the row it names is
    the first row of that layer that differs from its own single-row decode.
    """
    import torch

    reference = {index: value for index, value in reference_layers}
    candidate = {index: value for index, value in candidate_layers}
    differences = []
    for index in sorted(reference):
        expected = reference[index]
        actual = candidate.get(index)
        if actual is None:
            differences.append({"layer": index, "row": None, "missing": True})
            continue
        if tuple(expected.shape) != tuple(actual.shape):
            differences.append({"layer": index, "row": None,
                                "shape_mismatch": [list(expected.shape), list(actual.shape)]})
            continue
        for row in range(expected.shape[0]):
            if torch.equal(expected[row], actual[row]):
                continue
            entry = {"layer": index, "row": row,
                     "differing_elements": int((expected[row] != actual[row]).sum().item())}
            if expected.is_floating_point() and actual.is_floating_point():
                entry["max_abs"] = float(
                    (actual[row].float() - expected[row].float()).abs().max().item())
            differences.append(entry)
    return differences


def first_layer_row_difference(differences: list) -> dict | None:
    """The smallest layer index with any difference, and its first row."""
    ordered = sorted(
        differences,
        key=lambda entry: (entry.get("layer", 1 << 30),
                           -1 if entry.get("row") is None else entry["row"]),
    )
    return ordered[0] if ordered else None


def state_comparison(window, live) -> dict:
    """Compare the recorded rewind target with the live single-row state.

    `VerifyGraph.rewind` copies `window.to(live.dtype)` into the live buffer, so equality after that
    cast is the contract: a mismatch here means the rewind would restore a state plain decode never
    had, and every later step would decode from the wrong prefix.
    """
    if window is None or live is None:
        return {"equal": window is None and live is None,
                "window": None if window is None else "present",
                "live": None if live is None else "present"}
    cast = window.to(live.dtype) if window.dtype != live.dtype else window
    return compare_tensor(live, cast)


def ple_carry_slice(window, state_len: int, kept: int):
    """The PLE carry after keeping `kept` positions of the recorded window."""
    if kept < 0 or kept + state_len > window.shape[-1]:
        raise ValueError(
            f"PLE carry at {kept} does not fit a {window.shape[-1]}-wide window for {state_len}"
        )
    return window[..., kept:kept + state_len]


def ple_carry_comparison(window, live, state_len: int, kept: int) -> dict:
    """The recorded PLE window slice a rewind would keep, against single-row decode's carry."""
    return state_comparison(ple_carry_slice(window, state_len, kept), live)


class _CallRecorder:
    """A callable that records its result and delegates every attribute read to the real object."""

    def __init__(self, inner, sink) -> None:
        self.inner = inner
        self.sink = sink

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __call__(self, *args, **kwargs):
        value = self.inner(*args, **kwargs)
        self.sink(value)
        return value


def install_observers(model, store: dict, *, aux: bool) -> list:
    """Wrap every layer, and optionally its route/selection/injection calls, for recording.

    Wrapping the call objects rather than `F.linear`/`F.conv1d` keeps the model's arithmetic and
    every kernel untouched: the observer runs after the real call and only keeps the returned
    tensors. The return value of `install_observers` restores every attribute it changed.
    """
    saved = []

    def wrap(owner, name, sink):
        original = getattr(owner, name)
        saved.append((owner, name, original))
        setattr(owner, name, _CallRecorder(original, sink))

    for index, layer in enumerate(model.layers):
        if layer is None:
            continue
        if aux:
            if getattr(layer, "router", None) is not None:
                wrap(layer, "router", lambda value, index=index: store["router"].append(
                    (index, value[0], value[1])))
            attention = getattr(layer, "attn", None)
            if attention is not None and getattr(attention, "indexer", None) is not None:
                wrap(attention, "indexer", lambda value, index=index: store["indexer"].append(
                    (index, value)))
            if getattr(layer, "ple", None) is not None:
                wrap(layer, "ple", lambda value, index=index: store["ple"].append(
                    (index, value)))
        original = layer
        saved.append((model.layers, index, original))
        model.layers[index] = _CallRecorder(
            original, lambda value, index=index: store["layers"].append((index, value)))
    return saved


def restore_observers(saved: list) -> None:
    """Undo `install_observers`, newest first, tolerating a partially installed set."""
    for owner, name, value in reversed(saved):
        if isinstance(owner, list):
            owner[name] = value
        else:
            setattr(owner, name, value)
    saved.clear()


def reset_indexer_pools(cache) -> int:
    """Invalidate every request-owned indexer pool so both paths pool from their own raw keys.

    The pool is not part of the rewind snapshot: it is recomputed from position-addressed raw keys.
    Invalidating before each path is therefore exactness-preserving -- a pooled key is a pure
    function of the raw rows it covers -- and it stops the first path's pooled values from being
    read by the second, which would compare two paths that share a cache they did not write.
    """
    count = 0
    for layer in cache.layers:
        pool = getattr(layer, "indexer_pool", None) if layer is not None else None
        if pool is not None:
            pool.invalidate()
            count += 1
    return count


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--teacher-reference", required=True,
                        help="plain-artifact JSON/JSONL holding the validated output token list")
    parser.add_argument("--context-tokens", type=int, default=8192,
                        help="base context length built by the benchmark input helper")
    parser.add_argument("--deliver-prefix", type=int, default=64,
                        help="reference output tokens consumed one at a time before the window")
    parser.add_argument("--rows", type=int, default=6,
                        help="verify rows; the failing width-5 case is 6 verify rows")
    parser.add_argument("--chunk-size", type=int, default=512, help="base-context prefill chunk")
    parser.add_argument("--arena-slack", type=int, default=16)
    parser.add_argument("--label", default="")
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--skip-states", action="store_true",
                        help="skip recurrent/PLE state-log comparison (saves memory)")
    parser.add_argument("--skip-aux", action="store_true",
                        help="skip router/indexer/PLE output observers")
    parser.add_argument("--allow-input-mismatch", action="store_true",
                        help="continue even if the reference base-input hash differs")
    parser.add_argument("--allow-teacher-mismatch", action="store_true",
                        help="do not fail when the reproduced prefix disagrees with the reference")
    parser.add_argument("--tensor-past", action="store_true",
                        help="run the batch forward with a device-tensor past_len, the code path a "
                             "CUDA graph captures (serial steps stay eager integer positions)")
    return parser


def _ids(ids: list[int], device):
    import torch

    return torch.tensor([list(ids)], dtype=torch.long, device=device)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    validate_args(args)

    import torch
    import torch.distributed as dist

    from bench_qwen4_exp_baseline import _inputs
    from relicllm.models.qwen4_exp.model import restore_cache, snapshot_cache
    from relicllm.models.qwen4_exp.runtime import (
        _begin_ple_log, _begin_state_log, _end_state_log, gather_logits, init_distributed,
        load_model)

    ctx = init_distributed()
    if ctx.pp_size != 1 or ctx.device.type != "cuda":
        raise RuntimeError("this probe needs pure tensor parallelism on an explicitly leased GPU")
    evidence = (args.output_jsonl.open("x") if ctx.rank == 0 and args.output_jsonl else None)

    def emit(record: dict) -> None:
        if ctx.rank != 0:
            return
        line = json.dumps(record, sort_keys=True, allow_nan=False)
        print(line, flush=True)
        if evidence is not None:
            evidence.write(line + "\n")
            evidence.flush()

    teacher = load_teacher_reference(args.teacher_reference)
    window = select_window(teacher["ids"], args.deliver_prefix, args.rows)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True,
                                              trust_remote_code=True)
    base = _inputs(tokenizer, args.context_tokens)
    base_ids = [int(token) for token in base.reshape(-1)]
    base_hash = token_hash(base_ids)
    if (teacher["input_sha256"] is not None and teacher["input_sha256"] != base_hash
            and not args.allow_input_mismatch):
        raise SystemExit(
            "the teacher reference was produced from a different base input "
            f"(reference {teacher['input_sha256']}, rebuilt {base_hash}); pass "
            "--allow-input-mismatch only for a deliberate cross-context diagnostic"
        )
    model, config = load_model(args.model, ctx, expert_resident=True, expert_fused_fp8=True,
                               graph_window=0, mtp=False)
    native = int(config.text_config.max_position_embeddings)
    plan = plan_arena(args.context_tokens, args.deliver_prefix, args.rows, native_max=native,
                      slack=args.arena_slack)
    arena = plan["arena_tokens"]
    cache = model.make_cache(1, arena)
    if args.tensor_past:
        # The captured decode path reads fixed arena-width KV/rope state and derives positions from
        # a device tensor. Reproduce that metadata exactly, or the observation would describe a
        # different arithmetic path from the one the graph replays.
        with torch.inference_mode(False):
            cache.kv_width = arena
            cache.rope_width = arena
            for layer in cache.layers:
                if layer is not None:
                    layer.kv_width = arena
            if cache.cos is None or cache.cos.shape[1] < arena:
                positions = torch.arange(arena, device=ctx.device).view(1, 1, -1)
                cache.cos, cache.sin = model.rope(positions.expand(3, 1, -1), model.dtype)
            cache.share_rope_table()
    rows = args.rows
    emit({"kind": "qwen4_exp_all_rows_start", "schema": SCHEMA, "label": args.label,
          "model": args.model, "context_tokens": args.context_tokens,
          "deliver_prefix": args.deliver_prefix, "rows": rows, "chunk_size": args.chunk_size,
          "arena_tokens": arena, "requested_arena_tokens": plan["requested_arena_tokens"],
          "arena_clamped": plan["arena_clamped"], "logical_total_tokens": plan["logical_total_tokens"],
          "native_max_context": native, "base_input_sha256": base_hash,
          "base_input_hash_matches_reference":
              teacher["input_sha256"] is None or teacher["input_sha256"] == base_hash,
          "reference_path": teacher["path"], "reference_kind": teacher["source_kind"],
          "reference_output_tokens": len(teacher["ids"]),
          "feed_ids": window["feed_ids"], "teacher_predictions": window["teacher_predictions"],
          "flags": flag_snapshot(), "world_size": ctx.world_size,
          "tensor_past": bool(args.tensor_past), "measurement": "not_a_throughput_probe"})

    past = 0
    with torch.inference_mode():
        for piece in base.to(ctx.device).split(args.chunk_size, dim=1):
            model.forward(piece, cache=cache, past_len=past)
            past += piece.shape[1]
        for token in window["prefix_ids"]:
            model.forward(_ids([token], ctx.device), cache=cache, past_len=past)
            past += 1
    if past != args.context_tokens + args.deliver_prefix:
        raise RuntimeError(f"prefix reached {past} positions, not the requested teacher prefix")

    before = snapshot_cache(cache)
    batch_logits = None
    batch_states: list = []
    batch_ple_windows: list = []
    store = {"layers": [], "indexer": [], "router": [], "ple": []}
    saved = install_observers(model, store, aux=not args.skip_aux)
    try:
        reset_indexer_pools(cache)
        _begin_state_log(cache)
        if not args.skip_states:
            _begin_ple_log(cache)
        with torch.inference_mode():
            batch_position = (past if not args.tensor_past
                              else torch.tensor(past, dtype=torch.long, device=ctx.device))
            logits, _streams = model.forward(_ids(window["feed_ids"], ctx.device), cache=cache,
                                            past_len=batch_position, return_streams=True)
            batch_logits = gather_logits(logits[0], ctx).float().clone()
        batch_layers = {index: value.reshape(-1, value.shape[-1]).float()
                        for index, value in store["layers"]}
        batch_indexer = {index: value.reshape(rows, -1) for index, value in store["indexer"]}
        router_weights = {index: weights.reshape(rows, -1).float()
                          for index, weights, _ind in store["router"]}
        router_indices = {index: indices.reshape(rows, -1)
                          for index, _weights, indices in store["router"]}
        batch_ple = {index: value.reshape(rows, -1).float()
                     for index, value in store["ple"]}
        batch_states = [(index, list(getattr(layer, "state_log", None) or []))
                        for index, layer in enumerate(cache.layers)]
        batch_ple_windows = [list(getattr(state, "state_log", None) or [])
                             for state in (cache.ple_states or {}).values()]
    finally:
        restore_observers(saved)
        _end_state_log(cache, past)
        for state in (getattr(cache, "ple_states", None) or {}).values():
            state.state_log = None

    differences: list = []
    logits_differences: list = []
    teacher_disagreements: list = []
    batch_argmax = batch_logits.argmax(-1).reshape(-1)
    for row in range(rows):
        predicted = int(batch_argmax[row])
        if row < len(window["teacher_predictions"]) and predicted != window["teacher_predictions"][row]:
            teacher_disagreements.append({"path": "batch", "row": row, "argmax": predicted,
                                          "reference": window["teacher_predictions"][row]})

    restore_cache(before, cache)
    reset_indexer_pools(cache)
    seen = {"layers": [], "indexer": [], "router": [], "ple": []}
    saved = install_observers(model, seen, aux=not args.skip_aux)
    try:
        with torch.inference_mode():
            for row in range(rows):
                for key, bucket in seen.items():
                    bucket.clear()
                logits = model.forward(_ids([window["feed_ids"][row]], ctx.device), cache=cache,
                                       past_len=(past + row if not args.tensor_past
                                                 else torch.tensor(past + row, dtype=torch.long,
                                                                   device=ctx.device)))
                serial_logits = gather_logits(logits[0], ctx).float().clone()
                logits_differences.append({
                    "row": row, **compare_tensor(batch_logits[row], serial_logits[0])})
                serial_argmax = int(serial_logits[0].argmax())
                if (row < len(window["teacher_predictions"])
                        and serial_argmax != window["teacher_predictions"][row]):
                    teacher_disagreements.append({"path": "serial", "row": row,
                                                  "argmax": serial_argmax,
                                                  "reference": window["teacher_predictions"][row]})
                for index, value in seen["layers"]:
                    expected = batch_layers.get(index)
                    if expected is None:
                        continue
                    entries = layer_row_differences(
                        [(index, expected[row:row + 1])],
                        [(index, value.reshape(-1, value.shape[-1]).float())])
                    # `layer_row_differences` numbers rows inside the slice it was handed, so a
                    # one-row slice always reports row 0. Restore the row this serial step really
                    # occupies; otherwise every layer difference claims to be row zero.
                    for entry in entries:
                        entry["row"] = row
                    differences.extend(entries)
                for index, value in seen["indexer"]:
                    expected = batch_indexer.get(index)
                    if expected is not None:
                        result = compare_tensor(expected[row:row + 1], value.reshape(1, -1))
                        if not result["equal"]:
                            differences.append({"layer": index, "row": row, "call": "indexer",
                                                **result})
                for index, weights, indices in seen["router"]:
                    for name, expected, actual in (
                            ("router_weights", router_weights.get(index),
                             weights.reshape(1, -1).float()),
                            ("router_indices", router_indices.get(index),
                             indices.reshape(1, -1))):
                        if expected is None:
                            continue
                        result = compare_tensor(expected[row:row + 1], actual)
                        if not result["equal"]:
                            differences.append({"layer": index, "row": row, "call": name,
                                                **result})
                for index, value in seen["ple"]:
                    expected = batch_ple.get(index)
                    if expected is not None:
                        result = compare_tensor(
                            expected[row:row + 1], value.reshape(1, -1).float())
                        if not result["equal"]:
                            differences.append({"layer": index, "row": row, "call": "ple",
                                                **result})
                if not args.skip_states:
                    for index, layer in enumerate(cache.layers):
                        logged = dict(batch_states).get(index) or []
                        if index >= len(batch_states) or row >= len(logged):
                            continue
                        recorded_window, recorded_recurrent = logged[row]
                        for name, recorded, live in (
                                ("conv_state", recorded_window,
                                 getattr(layer, "conv_state", None)),
                                ("recurrent_state", recorded_recurrent,
                                 getattr(layer, "recurrent_state", None))):
                            result = state_comparison(recorded, live)
                            if not result["equal"]:
                                differences.append({"layer": index, "row": row, "call": name,
                                                    **result})
                    for index, state in enumerate((cache.ple_states or {}).values()):
                        windows = batch_ple_windows[index] if index < len(batch_ple_windows) else []
                        live = getattr(state, "conv_state", None)
                        if not windows or live is None:
                            continue
                        result = ple_carry_comparison(
                            windows[-1], live, int(live.shape[-1]), row + 1)
                        if not result["equal"]:
                            differences.append({"layer": index, "row": row, "call": "ple_carry",
                                                **result})
                cache.length = past + row + 1
                for layer in cache.layers:
                    if layer is not None and hasattr(layer, "length"):
                        layer.length = past + row + 1
    finally:
        restore_observers(saved)

    layer_differences = [entry for entry in differences if "call" not in entry]
    first = first_layer_row_difference(differences)
    local_summary = {"rank": ctx.rank, "different": bool(differences),
                     "layer_differences": len(layer_differences),
                     "logits_decoded_rows_differ": sum(
                         1 for entry in logits_differences if not entry["equal"]),
                     "teacher_disagreements": len(teacher_disagreements)}
    if dist.is_initialized():
        gathered = [None] * ctx.world_size
        dist.all_gather_object(gathered, local_summary)
    else:
        gathered = [local_summary]

    emit({"kind": "qwen4_exp_all_rows_result", "schema": SCHEMA, "label": args.label,
          "context_tokens": args.context_tokens, "deliver_prefix": args.deliver_prefix,
          "rows": rows, "positions": past,
          "batch_layers_recorded": len(batch_layers), "batch_ple_windows": len(batch_ple_windows),
          "layers_observed": len(batch_layers), "first_difference": first,
          "differences": differences[:64], "difference_count": len(differences),
          "layer_difference_count": len(layer_differences),
          "logits_differences": logits_differences, "teacher_disagreements": teacher_disagreements,
          "per_rank": gathered, "measurement": "not_a_throughput_probe"})
    failed = bool(differences) or bool(logits_differences and any(
        not entry["equal"] for entry in logits_differences))
    teacher_failed = bool(teacher_disagreements) and not args.allow_teacher_mismatch
    exit_code = 2 if failed else (3 if teacher_failed else 0)
    emit({"kind": "qwen4_exp_all_rows_summary", "schema": SCHEMA, "label": args.label,
          "different": bool(differences), "exit_code": exit_code,
          "rows_compared": rows, "first_difference": first,
          "teacher_disagreements": len(teacher_disagreements),
          "rate_measured": False,
          "interpretation": "a difference locates row-count-dependent arithmetic; it is not a "
                            "decode rate and not a claim that every row must be bit-identical "
                            "without the strict row-invariant flags",
          "absolute_reference_note": "teacher prefix was forced one token at a time, so a "
                                     "cache-arithmetic mismatch with the original run is reported "
                                     "as a teacher disagreement rather than hidden"})
    if evidence is not None:
        evidence.close()
    if dist.is_initialized():
        dist.barrier()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())