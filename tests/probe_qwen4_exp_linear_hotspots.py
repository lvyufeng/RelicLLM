"""Record the real shapes of Qwen4-Exp's decode-shaped dense projections, without changing them.

Why this exists. The main verifier is the dominant decode cost, and its small-batch projections run
as one `F.linear` per row precisely so their arithmetic matches single-token decode. Whether those
calls can be restructured is a question about the *actual* shapes and call sites in a real TP8
checkpoint, not about a guess from the config. This probe answers that question and nothing else: it
wraps the projection entry points, records metadata, and delegates to the original callable.

Two guarantees, both structural:

* **The arithmetic is the original arithmetic.** The wrappers call the original object and return
  its result unchanged. They never cast, copy, make contiguous, or select a different kernel; only
  host-readable metadata (`shape`, `dtype`, `stride()`, `is_contiguous()`) is inspected.
* **Steady decode is not instrumented.** After `on_prefill` the recorder stops accepting eager calls
  and only keeps calls that happen under CUDA graph capture. Captured work becomes a replay; what
  the recorder must enumerate is the prefill pass and the capture passes, which is exactly the set
  of Python-visible projection calls. A replay does not re-enter Python, so nothing per round is
  missed and nothing per round is slowed.

The private MTP head is quantized to NVFP4 **before any prefill**, matching the qualified arm: the
swap is one-way, and letting the first prefill run the FP8 head would record a different proposal
path.

Usage (observation pass, one MTP captured request, 8K in / 128 out)::

    python tests/probe_qwen4_exp_linear_hotspots.py --model <checkpoint> --out obs.json

Usage (serial-baseline micro from a saved observation; no model load, one card)::

    python tests/probe_qwen4_exp_linear_hotspots.py --micro-from obs.json --micro-out micro.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

SCHEMA = "qwen4-exp-linear-hotspots-v1"
MICRO_SCHEMA = "qwen4-exp-linear-hotspot-micro-v1"

#: Recorder phases: the chunked prompt pass, then everything after it (capture setup).
PHASE_PREFILL = "prefill"
PHASE_DECODE_SETUP = "decode_setup"


# --------------------------------------------------------------------------------------------
# Tensor-free metadata extraction. These run on duck-typed objects, so the recorder's logic is
# unit-testable without torch, a checkpoint, or a card.
# --------------------------------------------------------------------------------------------


def _tensor_signature(value) -> dict | None:
    """`shape`, `dtype`, `strides` and contiguity of a tensor-like object, or ``None``."""
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        shape = tuple(int(dimension) for dimension in shape)
    except (TypeError, ValueError):
        return None
    dtype = getattr(value, "dtype", None)
    strides = None
    try:
        strides = tuple(int(stride) for stride in value.stride())
    except (AttributeError, TypeError, ValueError):
        strides = None
    contiguous = None
    try:
        contiguous = bool(value.is_contiguous())
    except (AttributeError, TypeError, ValueError):
        contiguous = None
    return {"shape": list(shape), "dtype": str(dtype), "strides": strides,
            "contiguous": contiguous}


def describe_linear(x, weight, bias=None) -> dict | None:
    """The projection geometry of one `F.linear`-shaped call, or ``None`` if it is not one.

    ``rows`` is every leading element of ``x`` folded together, the count the row-invariant path
    loops over. `in_features`/`out_features` come from the weight, so a transposed or unusual weight
    is visible in the recorded signature instead of being silently assumed.
    """
    x_signature = _tensor_signature(x)
    weight_signature = _tensor_signature(weight)
    if x_signature is None or weight_signature is None:
        return None
    if len(x_signature["shape"]) < 1 or len(weight_signature["shape"]) != 2:
        return None
    out_features, in_features = weight_signature["shape"]
    hidden = x_signature["shape"][-1]
    rows = 1
    for dimension in x_signature["shape"][:-1]:
        rows *= dimension
    return {
        "rows": int(rows),
        "in_features": int(in_features),
        "out_features": int(out_features),
        "x_hidden": int(hidden),
        "x_ndim": len(x_signature["shape"]),
        "x_shape": x_signature["shape"],
        "x_dtype": x_signature["dtype"],
        "x_strides": x_signature["strides"],
        "x_contiguous": x_signature["contiguous"],
        "weight_shape": weight_signature["shape"],
        "weight_dtype": weight_signature["dtype"],
        "weight_strides": weight_signature["strides"],
        "weight_contiguous": weight_signature["contiguous"],
        # A `False` here means the call does not take its hidden dimension from the weight's
        # columns; recorded rather than raised, because the census must describe what ran.
        "hidden_matches_weight": int(hidden) == int(in_features),
        "has_bias": bias is not None,
    }


def caller_site(frame, root: Path | str = ROOT) -> str:
    """A stable ``relative/path.py:line:function`` label for the frame that issued the call."""
    filename = frame.f_code.co_filename
    try:
        relative = os.path.relpath(filename, str(root))
    except ValueError:  # different drives on a Windows host; keep the absolute path
        relative = filename
    if relative.startswith(".."):
        relative = filename
    return f"{relative}:{frame.f_lineno}:{frame.f_code.co_name}"


def _is_capturing() -> bool:
    """Whether the current CUDA stream is inside a graph capture; always False without CUDA."""
    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except (RuntimeError, AssertionError):
        return False


class ProjectionRecorder:
    """Aggregate projection calls by call site and geometry; accept, never alter, the call.

    One row per ``(kind, site, geometry, phase, capturing)``. A call nested inside
    ``row_invariant_linear`` is counted on its parent's entry as ``serial_children`` instead of
    becoming a second top-level projection, because the row loop *is part of* that projection.
    """

    def __init__(self, *, root: Path | str = ROOT, max_entries: int = 4096) -> None:
        self.root = str(root)
        self.max_entries = int(max_entries)
        self.phase = PHASE_PREFILL
        self.entries: dict[tuple, dict] = {}
        self.stack: list[tuple] = []
        self.skipped_steady = 0
        self.skipped_capacity = 0
        self.malformed = 0

    # -- control ---------------------------------------------------------------------------

    def set_phase(self, phase: str) -> None:
        if phase not in (PHASE_PREFILL, PHASE_DECODE_SETUP):
            raise ValueError(f"unknown phase {phase!r}")
        self.phase = phase

    def enabled(self, *, capturing: bool) -> bool:
        """Prefill and capture are recorded; steady eager decode is deliberately not."""
        return self.phase == PHASE_PREFILL or capturing

    # -- recording -------------------------------------------------------------------------

    def record(self, *, kind: str, args, kwargs, frame, capturing: bool):
        """Record one intercepted call and return the key it opened, or ``None``.

        Returning the opened key (rather than tracking depth in a separate field) is what keeps the
        nesting correct: a nested call opens nothing, so its ``finally`` must not pop its parent.
        """
        if not self.enabled(capturing=capturing):
            self.skipped_steady += 1
            return None
        try:
            positional = list(args)
            if len(positional) < 2 and len(kwargs) >= 2:
                positional = [kwargs.get("input"), kwargs.get("weight"), kwargs.get("bias")]
            if len(positional) < 2:
                self.malformed += 1
                return None
            geometry = describe_linear(positional[0], positional[1],
                                       positional[2] if len(positional) > 2 else None)
        except Exception:  # a malformed call must not fail the request that made it
            self.malformed += 1
            return None
        if geometry is None:
            self.malformed += 1
            return None
        if self.stack:
            parent = self.entries.get(self.stack[-1])
            if parent is not None:
                parent["serial_children"] += 1
            return None
        site = caller_site(frame, self.root)
        key = (kind, site, geometry["rows"], geometry["in_features"],
               geometry["out_features"], geometry["x_dtype"], geometry["weight_dtype"],
               geometry["x_ndim"], geometry["x_contiguous"], geometry["weight_contiguous"],
               self.phase, capturing, geometry["has_bias"])
        entry = self.entries.get(key)
        if entry is None:
            if len(self.entries) >= self.max_entries:
                self.skipped_capacity += 1
                return None
            entry = {"kind": kind, "site": site, "phase": self.phase, "capturing": capturing,
                     "calls": 0, "serial_children": 0, **geometry}
            self.entries[key] = entry
        entry["calls"] += 1
        self.stack.append(key)
        return key

    def finish_call(self, key) -> None:
        """Close the entry a wrapper opened; a nested call opened nothing and closes nothing."""
        if key is not None and self.stack and self.stack[-1] == key:
            self.stack.pop()

    # -- reporting -------------------------------------------------------------------------

    def document(self) -> dict:
        entries = [dict(entry) for entry in self.entries.values()]
        entries.sort(key=lambda entry: (-entry["calls"], entry["kind"], entry["site"]))
        hot: dict[tuple, dict] = {}
        for entry in entries:
            key = (entry["rows"], entry["in_features"], entry["out_features"],
                   entry["weight_dtype"], entry["x_ndim"])
            row = hot.get(key)
            if row is None:
                row = {"rows": entry["rows"], "in_features": entry["in_features"],
                       "out_features": entry["out_features"],
                       "weight_dtype": entry["weight_dtype"], "x_ndim": entry["x_ndim"],
                       "calls": 0, "sites": []}
                hot[key] = row
            row["calls"] += entry["calls"]
            if entry["site"] not in row["sites"]:
                row["sites"].append(entry["site"])
        hot_rows = sorted(hot.values(),
                          key=lambda row: (-row["calls"],
                                           -row["rows"] * row["in_features"]))
        for row in hot_rows:
            row["sites"].sort()
            itemsize = 2 if ("bfloat16" in row["weight_dtype"]
                             or "float16" in row["weight_dtype"]) else 4
            row["weight_bytes_one_call"] = (row["out_features"] * row["in_features"] * itemsize)
        phases: dict[str, int] = {}
        for entry in entries:
            phases[entry["phase"]] = phases.get(entry["phase"], 0) + entry["calls"]
        return {
            "entries": entries,
            "hot_projection_shapes": hot_rows,
            "calls_by_phase": phases,
            "calls_total": sum(entry["calls"] for entry in entries),
            "serial_children_total": sum(entry["serial_children"] for entry in entries),
            "skipped_steady_calls": self.skipped_steady,
            "skipped_capacity": self.skipped_capacity,
            "malformed_calls": self.malformed,
        }


class observe_projection_calls:
    """Temporarily wrap the projection entry points with a :class:`ProjectionRecorder`.

    Installed: `relicllm.models.qwen4_exp.layers.row_invariant_linear`,
    `relicllm.models.qwen4_exp.model.row_invariant_linear` (a separate module-level binding, so both
    attributes need the wrapper), and `torch.nn.functional.linear`. Each wrapper returns the original
    call's own result. Restoration is in ``finally`` and puts the original objects back, so a failed
    request cannot leave instrumentation installed.
    """

    def __init__(self, recorder: ProjectionRecorder) -> None:
        self.recorder = recorder
        self._restore: list[tuple[object, str, object]] = []

    def __enter__(self) -> ProjectionRecorder:
        import torch.nn.functional as functional
        from relicllm.models.qwen4_exp import layers
        from relicllm.models.qwen4_exp import model as model_module

        recorder = self.recorder

        def wrap(kind: str, original):
            def wrapper(*args, **kwargs):
                frame = sys._getframe(1)
                capturing = _is_capturing()
                key = recorder.record(kind=kind, args=args, kwargs=kwargs,
                                      frame=frame, capturing=capturing)
                try:
                    return original(*args, **kwargs)
                finally:
                    recorder.finish_call(key)
            wrapper.__name__ = f"observed_{kind}"
            wrapper.__doc__ = f"Record-only wrapper around {kind}; delegates the call unchanged."
            return wrapper

        for module, attribute in ((layers, "row_invariant_linear"),
                                  (model_module, "row_invariant_linear")):
            current = getattr(module, attribute)
            self._restore.append((module, attribute, current))
            setattr(module, attribute, wrap("row_invariant_linear", current))
        original_functional_linear = functional.linear
        self._restore.append((functional, "linear", original_functional_linear))
        setattr(functional, "linear", wrap("F_linear", original_functional_linear))
        return recorder

    def __exit__(self, exc_type, exc, tb) -> bool:
        for target, attribute, original in reversed(self._restore):
            setattr(target, attribute, original)
        self._restore = []
        return False


# --------------------------------------------------------------------------------------------
# Serial-baseline micro. Deliberately one candidate: the production row-by-row arithmetic.
# --------------------------------------------------------------------------------------------


def serial_row_tensor(flat, row: int, x_ndim: int):
    """The tensor `row_invariant_linear` hands `F.linear` for one row, exactly.

    The model flattens the batch to `(rows, in)`, then hands `F.linear` a tensor whose leading
    dimensions are all one, `(1,) * (x_ndim - 1) + (in,)`. That leading shape is why a decode-shaped
    call is a GEMV while a prefill chunk is a GEMM; the micro restores it rather than measuring a
    different geometry.
    """
    single_shape = (1,) * (int(x_ndim) - 1) + (flat.shape[-1],)
    return flat[row:row + 1].reshape(single_shape).contiguous()


def micro_serial_baseline(shapes: list[dict], *, device_index: int, iterations: int,
                          seed: int = 20261119) -> dict:
    """Time the production serial path per shape. No batched or bmm variant is measured."""
    import torch
    import torch.nn.functional as functional

    if not torch.cuda.is_available():
        raise RuntimeError("the serial micro benchmark needs a CUDA device")
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)
    torch.manual_seed(seed)
    results = []
    for shape in shapes:
        rows = int(shape["rows"])
        in_features = int(shape["in_features"])
        out_features = int(shape["out_features"])
        x_ndim = int(shape.get("x_ndim", 2))
        wanted = str(shape.get("weight_dtype", "torch.bfloat16")).removeprefix("torch.")
        # bfloat16 contains the substring float16: exact names keep the production dtype intact.
        if wanted == "bfloat16":
            dtype = torch.bfloat16
        elif wanted == "float16":
            dtype = torch.float16
        elif wanted == "float32":
            dtype = torch.float32
        else:
            raise ValueError(f"unsupported observed weight dtype: {wanted}")
        weight = torch.randn(out_features, in_features, device=device, dtype=dtype)
        flat = torch.randn(rows, in_features, device=device, dtype=dtype)

        def serial_rows():
            for row in range(rows):
                functional.linear(serial_row_tensor(flat, row, x_ndim), weight)

        def one_row():
            functional.linear(serial_row_tensor(flat, 0, x_ndim), weight)

        def timed(call):
            for _ in range(5):
                call()
            torch.cuda.synchronize(device)
            samples = []
            for _ in range(iterations):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                call()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))
            return samples

        row_samples = timed(serial_rows)
        one_samples = timed(one_row)
        results.append({
            "rows": rows, "in_features": in_features, "out_features": out_features,
            "x_ndim": x_ndim, "dtype": str(dtype).replace("torch.", ""),
            "serial_rows_ms_median": round(statistics.median(row_samples), 6),
            "serial_rows_ms_min": round(min(row_samples), 6),
            "one_row_ms_median": round(statistics.median(one_samples), 6),
            "iterations": iterations,
        })
        del weight, flat
        torch.cuda.empty_cache()
    return {
        "schema": MICRO_SCHEMA,
        "kind": "qwen4_exp_linear_hotspot_micro",
        "serial_only": True,
        "note": ("Times only the production row-by-row F.linear sequence, including its per-row "
                 "leading-singleton shape. No bmm or batched replacement is measured: a batched form "
                 "is a separately reviewed change that must prove bit-identical output over mutated "
                 "inputs, not equal output on one tensor. The final torch.cat is not included."),
        "device": str(device),
        "results": results,
    }


def micro_shapes(document: dict) -> list[dict]:
    """The unique projection geometries of a saved observation document."""
    if document.get("schema") != SCHEMA:
        raise ValueError(f"micro needs a {SCHEMA} observation document")
    shapes = document.get("hot_projection_shapes")
    if not isinstance(shapes, list) or not shapes:
        raise ValueError("observation document has no hot_projection_shapes")
    unique: dict[tuple, dict] = {}
    for shape in shapes:
        try:
            key = (int(shape["rows"]), int(shape["in_features"]),
                   int(shape["out_features"]), str(shape["weight_dtype"]),
                   int(shape.get("x_ndim", 2)))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed hot projection shape: {shape!r}") from exc
        if key[0] < 1 or key[1] < 1 or key[2] < 1 or key[4] < 1:
            raise ValueError(f"a hot projection shape must be positive: {shape!r}")
        unique[key] = {"rows": key[0], "in_features": key[1], "out_features": key[2],
                       "weight_dtype": key[3], "x_ndim": key[4]}
    return list(unique.values())


# --------------------------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------------------------


def _write_exclusive(path: Path, document: dict) -> None:
    """Write a document only if the path is new: an evidence file is never silently replaced."""
    if path.exists():
        raise FileExistsError(f"{path} already exists; name a new evidence file")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(document, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _observe(args) -> int:
    if os.environ.get("POCKETLLM_QWEN4_ROUND_TRACE"):
        raise RuntimeError("do not census shapes under a round trace; those are separate passes")
    if os.environ.get("POCKETLLM_QWEN4_MTP_NVFP4", "0") != "1":
        raise RuntimeError("set POCKETLLM_QWEN4_MTP_NVFP4=1 so the recorded request matches the "
                           "qualified NVFP4-head arm instead of the one-way FP8 head")
    import torch
    import torch.distributed as dist
    from transformers import AutoTokenizer

    from bench_qwen4_exp_baseline import _inputs
    from probe_qwen4_mtp_nvfp4 import (_checkpoint_identity, _kernel_identity,
                                       _math_environment, _source_hashes, _token_hash)
    from relicllm.models.qwen4_exp.runtime import (_warmup_device, generate_mtp_draft_lookahead,
                                                  init_distributed, load_model)
    from relicllm.models.qwen4_exp.verify_graph import _quantize_head_experts

    ctx = init_distributed()
    if ctx.pp_size != 1 or ctx.device.type != "cuda":
        raise RuntimeError("requires pure TP on an explicitly leased CUDA host")

    checkpoint = Path(args.model)
    model, config = load_model(args.model, ctx, expert_resident=True, expert_fused_fp8=True,
                               graph_window=args.graph_window, mtp=True)
    if not model.has_mtp:
        raise RuntimeError("the model was built without its MTP head")
    native = int(config.text_config.max_position_embeddings)
    budgets = {layer.attn.indexer.budget for layer in model.layers
               if layer is not None and hasattr(layer.attn, "indexer")}
    if budgets != {2048}:
        raise RuntimeError(f"the effective indexer budget must remain 2048, got {sorted(budgets)}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True,
                                              trust_remote_code=True)
    inputs = _inputs(tokenizer, args.context_tokens)
    ids = [int(token) for token in inputs[0].tolist()]
    if inputs.ndim != 2 or inputs.shape[0] != 1 or len(ids) != args.context_tokens:
        raise RuntimeError("the constructed input does not have the exact requested token count")

    # Before ANY prefill: the head's FP8->NVFP4 swap is one-way and prefill already produces the
    # first draft through the head, so quantizing afterwards would record a different proposal path.
    _quantize_head_experts(model.mtp.moe, ctx.device)
    head_nvfp4 = type(model.mtp.moe.inner).__name__ == "NvFp4SlotExpertMoE"
    if not head_nvfp4:
        raise RuntimeError("the private head did not become an NVFP4 slot MoE")

    recorder = ProjectionRecorder(root=ROOT)
    warmup_s = _warmup_device(ctx=ctx, dtype=model.dtype, seconds=args.warmup_s)
    phases: list[str] = [PHASE_PREFILL]

    def enter_decode_setup(_state=None) -> None:
        recorder.set_phase(PHASE_DECODE_SETUP)
        phases.append(PHASE_DECODE_SETUP)

    with observe_projection_calls(recorder):
        output, stats = generate_mtp_draft_lookahead(
            model, ctx, inputs, max_new_tokens=args.max_new_tokens,
            chunk_size=args.chunk_size, draft_tokens=args.draft_tokens,
            verify_graph=True, eos_token_ids=(), on_prefill=enter_decode_setup,
        )
    torch.cuda.synchronize(ctx.device)

    document = recorder.document()
    ranks = [None] * ctx.world_size
    local = {"rank": ctx.rank,
             "hot": [[row["rows"], row["in_features"], row["out_features"], row["x_ndim"]]
                     for row in document["hot_projection_shapes"]],
             "calls_total": document["calls_total"]}
    dist.all_gather_object(ranks, local)
    same_hot = all(rank is not None
                   and rank["hot"] == ranks[0]["hot"] for rank in ranks)
    same_total = all(rank is not None
                     and rank["calls_total"] == ranks[0]["calls_total"] for rank in ranks)

    payload = {
        "kind": "qwen4_exp_linear_hotspot_observation",
        "schema": SCHEMA,
        "scope": ("metadata census of the projections a real TP8 checkpoint calls in prefill and "
                  "capture setup; every call delegates unchanged; no throughput claim"),
        "measurement_scope": {
            "recording_window": ("the eager prefill pass plus CUDA-graph capture passes; steady "
                                 "eager decode is skipped because a captured replay does not "
                                 "re-enter Python"),
            "phases_seen": phases,
            "skipped_steady_calls": document["skipped_steady_calls"],
            "skipped_capacity": document["skipped_capacity"],
        },
        "traffic_caveat": (
            "weight_bytes_one_call is out_features*in_features*itemsize for ONE call of a shape, not "
            "measured HBM traffic. The rows of one projection are issued back to back, so a matrix "
            "that fits in the card's L2 can already be reused across them; do not multiply the whole "
            "model's dense bytes by a row count and call that DRAM traffic."),
        "request": {
            "context_tokens": args.context_tokens,
            "max_new_tokens": args.max_new_tokens,
            "chunk_size": args.chunk_size,
            "draft_tokens": args.draft_tokens,
            "verify_graph": True,
            "plain_graph_window": args.graph_window,
            "eos_token_ids": [],
            "indexer_budget": 2048,
            "native_max_position_embeddings": native,
            "model_directory": str(checkpoint.resolve()),
        },
        "head": {"nvfp4_prequantized_before_prefill": True,
                 "nvfp4_requested_by_env": True,
                 "inner_class": type(model.mtp.moe.inner).__name__},
        "identity": {
            "checkpoint": _checkpoint_identity(checkpoint),
            "sources": _source_hashes(),
            "kernel": _kernel_identity(),
            "math_environment": _math_environment(),
            "world_size": ctx.world_size,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "warmup_s": warmup_s,
        },
        "observer": {
            "calls_by_phase": document["calls_by_phase"],
            "calls_total": document["calls_total"],
            "serial_children_total": document["serial_children_total"],
            "malformed_calls": document["malformed_calls"],
            "entries": len(document["entries"]),
        },
        "all_ranks_same_hot_shapes": same_hot,
        "all_ranks_same_call_total": same_total,
        "per_rank": ranks,
        "entries": document["entries"],
        "hot_projection_shapes": document["hot_projection_shapes"],
        "decode": {"produced_tokens": len(output), "output_sha256": _token_hash(output),
                   "decode_tps_reported": stats.get("decode_tps"),
                   "note": "the rate is from an instrumented metadata pass, not a TPS result"},
    }
    if ctx.rank == 0:
        _write_exclusive(args.out, payload)
        print(json.dumps({"kind": "qwen4_exp_linear_hotspot_summary", "schema": SCHEMA,
                          "out": str(args.out.resolve()),
                          "calls_total": document["calls_total"],
                          "hot_projection_shapes": len(document["hot_projection_shapes"]),
                          "all_ranks_same": same_hot and same_total,
                          "produced_tokens": len(output)}, sort_keys=True), flush=True)
    return 0 if (same_hot and same_total) else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="checkpoint directory (required for the observation pass)")
    parser.add_argument("--out", type=Path, help="write the observation document here")
    parser.add_argument("--context-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--draft-tokens", type=int, default=3)
    parser.add_argument("--graph-window", type=int, default=32)
    parser.add_argument("--warmup-s", type=float, default=1.0)
    parser.add_argument("--micro-from", type=Path,
                        help="time the serial baseline for a saved observation document instead")
    parser.add_argument("--micro-out", type=Path)
    parser.add_argument("--micro-iters", type=int, default=20)
    parser.add_argument("--device-index", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.micro_from is not None:
        if args.micro_out is None:
            raise SystemExit("--micro-from also needs --micro-out")
        if args.model is not None:
            raise SystemExit("--micro-from measures saved shapes only; do not also load a model")
        if args.micro_iters < 1:
            raise SystemExit("--micro-iters must be positive")
        document = json.loads(args.micro_from.read_text())
        result = micro_serial_baseline(micro_shapes(document),
                                       device_index=args.device_index,
                                       iterations=args.micro_iters)
        _write_exclusive(args.micro_out, result)
        print(json.dumps({"kind": "qwen4_exp_linear_hotspot_micro_summary",
                          "schema": MICRO_SCHEMA, "out": str(args.micro_out.resolve()),
                          "shapes": len(result["results"])}, sort_keys=True), flush=True)
        return 0
    if args.model is None or args.out is None:
        raise SystemExit("the observation pass needs --model and --out")
    if args.context_tokens < 1 or args.max_new_tokens < 2 or args.chunk_size < 1:
        raise SystemExit("context tokens and chunk size must be positive; two outputs minimum")
    return _observe(args)


if __name__ == "__main__":
    sys.exit(main())