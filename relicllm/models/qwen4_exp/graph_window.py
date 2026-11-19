"""One position-independent decode recording with transactional cache ownership.

A recorded step reads one GDN/PLE state slot and writes the next. Slot zero is
seeded from the accepted prefix, so warmup and capture may write slot one without
advancing that prefix. A successful replay commits its output slot; a declined
replay releases only the last committed state before the caller resumes eagerly.

KV arenas are not cloned: setup and failed replays write only uncommitted tail
positions, which restored logical lengths hide until the next forward overwrites
them. Widths, rotary-table references, history hooks and installed state layouts
are request-local and restored when the recording gives ownership back.

PLE hashing and host-table lookup do not run during CUDA replay. Each recording
owns fixed pinned row buffers: fill_history refreshes them on the CPU, and the
recorded H2D copies read those exact addresses. Shared table scratch buffers are
bound only during setup and restored before other requests may use the model.

These lifecycle invariants do not guarantee numerical or token parity with eager
CUDA execution. That requires a device-side logit/state oracle; a return margin
is only an optional handback policy, not proof of equivalence.
"""
from __future__ import annotations

import os

import torch

from .attention import GatedDeltaNetCache, slot_layout


_MISSING = object()
_ROOT_ATTRIBUTES = (
    "kv_width", "rope_width", "ple_history_at", "length", "ple_token_history",
    "cos", "sin", "_decode_graph_owner",
)
_LAYER_ATTRIBUTES = (
    "kv_width", "rope_width", "cos", "sin", "length", "host_position",
    "state_slots", "state_slot", "initialized", "state_log",
    "conv_state", "recurrent_state",
)


class GraphLifecycleError(RuntimeError):
    """The caller must release existing graph state before changing its ownership."""


class GraphRollbackError(RuntimeError):
    """Cache/device restoration failed; continuing with eager execution is unsafe."""


def _return_margin() -> float:
    """The relative top-2 gap below which a replay hands the step back to eager."""
    try:
        return float(os.environ.get("POCKETLLM_GRAPH_RETURN_MARGIN", "0"))
    except ValueError:
        return 0.0


class _Attributes:
    """Save references and attribute presence, not copies of large tensor arenas."""

    def __init__(self, objects) -> None:
        self.saved = []
        for target, names in objects:
            namespace = getattr(target, "__dict__", None)
            values = {}
            for name in names:
                # Preserve instance-attribute absence even when a class has a default.
                if namespace is not None:
                    values[name] = namespace.get(name, _MISSING)
                else:
                    values[name] = getattr(target, name, _MISSING)
            self.saved.append((target, values))

    def value(self, target, name):
        for obj, values in self.saved:
            if obj is target:
                return values.get(name, _MISSING)
        return _MISSING

    def restore(self, names=None) -> None:
        for target, values in self.saved:
            for name, value in values.items():
                if names is not None and name not in names:
                    continue
                if value is _MISSING:
                    namespace = getattr(target, "__dict__", None)
                    present = name in namespace if namespace is not None else hasattr(target, name)
                    if present:
                        delattr(target, name)
                else:
                    setattr(target, name, value)


class _CacheTransaction:
    """Freeze only live recurrent/conv buffers and the tiny host token prefix."""

    def __init__(self, cache, bindings) -> None:
        self.cache = cache
        self.bindings = bindings
        objects = [(cache, _ROOT_ATTRIBUTES)]
        seen = {id(cache)}
        for target in [*cache.layers, *(target for target, _ in bindings)]:
            if target is not None and id(target) not in seen:
                seen.add(id(target))
                objects.append((target, _LAYER_ATTRIBUTES))
        self.attributes = _Attributes(objects)
        self.frozen = []
        with torch.inference_mode(False):
            for target, fields in bindings:
                self.frozen.append({name: getattr(target, name).clone() for name in fields})
            history = self.attributes.value(cache, "ple_token_history")
            self.history = history.clone() if isinstance(history, torch.Tensor) else history

    def _copy_states(self, values=None) -> None:
        for index, (target, fields) in enumerate(self.bindings):
            source = self.frozen[index] if values is None else values[index]
            for name in fields:
                live = self.attributes.value(target, name)
                value = source[name]
                if not isinstance(live, torch.Tensor) or live.shape != value.shape:
                    raise RuntimeError(f"cannot restore {name} into its original live buffer")
                live.copy_(value)

    def _copy_history(self, history) -> None:
        live = self.attributes.value(self.cache, "ple_token_history")
        if isinstance(live, torch.Tensor) and isinstance(history, torch.Tensor):
            if live.shape != history.shape:
                raise RuntimeError("cannot restore token history into its original buffer")
            live.copy_(history)

    def restore_logical_prefix(self) -> None:
        """Undo setup's logical effects while keeping the installed recording layout."""
        with torch.inference_mode():
            self._copy_states()
            self._copy_history(self.history)
        self.attributes.restore({
            "length", "ple_token_history", "initialized", "host_position",
            "conv_state", "recurrent_state",
        })

    def restore(self, *, values=None, length=None, history=_MISSING, committed=False) -> None:
        prefix = self.history if history is _MISSING else history
        with torch.inference_mode():
            self._copy_states(values)
            self._copy_history(prefix)
        self.attributes.restore()
        if length is not None:
            for target, attributes in self.attributes.saved:
                if attributes.get("length", _MISSING) is not _MISSING:
                    target.length = int(length)
            if committed:
                for target, _fields in self.bindings:
                    if self.attributes.value(target, "initialized") is not _MISSING:
                        target.initialized = True
            original = self.attributes.value(self.cache, "ple_token_history")
            if original is not _MISSING:
                if isinstance(original, torch.Tensor) and isinstance(prefix, torch.Tensor):
                    self.cache.ple_token_history = original
                elif isinstance(prefix, torch.Tensor):
                    self.cache.ple_token_history = prefix.clone()
                else:
                    self.cache.ple_token_history = prefix


class DecodeGraph:
    """Record one step, commit accepted slots, and release cache ownership explicitly."""

    def __init__(self, *, model, cache, start: int, steps: int, token: torch.Tensor,
                 select=None, wants_streams: bool = False) -> None:
        self.model = model
        self.cache = cache
        self.start = int(start)
        self.steps = int(steps)
        self.select = select
        self.wants_streams = bool(wants_streams)
        self.graph = None
        self._transaction = None
        self._layouts = []
        self._ple_rows = []
        self._table_attributes = None
        self._history_ready = False
        self._released = False
        self._restore_error = None
        self._committed_slot = 0
        self._committed_any = False
        self.reuse_windows = os.environ.get("POCKETLLM_QWEN4_GRAPH_REUSE", "0") == "1"
        self.rebase_count = 0
        self.rebase_s = 0.0
        self._committed_length = self.start
        self._committed_history = _MISSING
        self._device = token.device
        self.batch = int(token.shape[0]) if token.ndim > 0 else 0
        self._ngram = int(model.config.ngram_size)
        self._validate_token(token)
        if self.steps <= 0 or self.start < 0 or self._ngram <= 0:
            raise ValueError("a decode graph needs positive steps/ngram size and a nonnegative start")
        if getattr(model, "pp_size", 1) > 1 or getattr(model, "lm_head", None) is None:
            raise ValueError("decode graphs require a non-pipeline model with a local head")
        if getattr(cache, "_decode_graph_owner", None) is not None:
            raise GraphLifecycleError("release the existing decode graph before building another")
        if getattr(cache, "length", None) != self.start:
            raise ValueError("graph start must equal the cache's committed length")

        self._bindings = []
        for layer in cache.layers:
            if isinstance(layer, GatedDeltaNetCache):
                self._bindings.append((layer, ("conv_state", "recurrent_state")))
        ple_states = getattr(cache, "ple_states", {})
        for state in ple_states.values():
            self._bindings.append((state, ("conv_state",)))
        for target, fields in self._bindings:
            if getattr(target, "state_slots", None) is not None:
                raise GraphLifecycleError("release the existing state-slot layout before capture")
            for name in fields:
                value = getattr(target, name, None)
                if (not isinstance(value, torch.Tensor) or value.ndim == 0
                        or value.shape[0] != self.batch):
                    raise ValueError(f"{name} must be a fixed tensor for this graph's batch")
        for state in ple_states.values():
            span = (int(model.config.ple_conv_kernel_size) - 1) * self._ngram
            if state.conv_state.shape != (self.batch, self._streams_width(), span):
                raise ValueError("PLE conv_state must have shape [batch, hc_hidden, state_len]")

        try:
            self._transaction = _CacheTransaction(cache, self._bindings)
            self._committed_history = self._transaction.history
            self.width = self._arena_len()
            # CPU allocation is useful only with a mocked capture in lifecycle tests;
            # production build() still requires CUDA availability.
            pinned = self._device.type == "cuda"
            with torch.inference_mode(False):
                self.inputs = token.clone()
                self._host_token = torch.zeros(self.batch, 1, dtype=torch.long, pin_memory=pinned)
                self._host_token.copy_(token)
                self.position = torch.full((), self.start, dtype=torch.long, device=self._device)
                self.slot = torch.zeros((), dtype=torch.long, device=self._device)
                self.history = torch.full(
                    (self.batch, self._ngram), int(model.config.primary_eos_token_id),
                    dtype=torch.long, pin_memory=pinned,
                )
                if isinstance(self._committed_history, torch.Tensor):
                    if self._committed_history.shape != (self.batch, self._ngram - 1):
                        raise ValueError("the committed PLE token prefix has an unexpected shape")
                    self.history[:, :-1].copy_(self._committed_history)
                self.history[:, -1:].copy_(self._host_token)
                self.outputs = torch.zeros(
                    self.batch, self._logits_width(), dtype=torch.float32, device=self._device
                )
                self.streams_buf = (
                    torch.zeros(self.batch, self._streams_width(), dtype=torch.float32,
                                device=self._device) if self.wants_streams else None
                )
            self._prepare()
            self._capture()
            self._synchronize()
            if self.graph is None:
                raise RuntimeError("capture did not produce a replayable graph")
            for binding in self._ple_rows:
                if binding["table"]._rows_buffer is not binding["buffer"]:
                    raise RuntimeError("PLE table replaced the graph's fixed host-row source")
            self._restore_table_bindings()
            self._transaction.restore_logical_prefix()
        except BaseException:
            # Includes an interrupted warmup/capture. A failed rollback propagates a
            # distinct error so build() cannot turn a poisoned device into eager work.
            self.release()
            raise

    def _validate_token(self, token: torch.Tensor) -> None:
        if token.shape != (self.batch, 1) or self.batch <= 0 or token.dtype != torch.long:
            raise ValueError(f"decode graph tokens must be int64 [{self.batch}, 1]")

    def _synchronize(self) -> None:
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)

    def _arena_len(self) -> int:
        """Read capacity without probing a forward or allocating another KV arena."""
        reach = self.start + self.steps + 2
        widths = []
        for layer in self.cache.layers:
            if layer is None:
                continue
            for name in ("k", "v", "index_k"):
                tensor = getattr(layer, name, None)
                if tensor is not None:
                    width = int(tensor.shape[2] if tensor.ndim == 4 else tensor.shape[1])
                    if width < self.start + self.steps:
                        raise ValueError("the graph span exceeds the caller's KV arena")
                    widths.append(width)
        width = max([reach, *widths])
        self._replay_limit = min([width, *widths])
        return width

    def _streams_width(self) -> int:
        cfg = self.model.config
        return int(getattr(cfg, "hc_count", 1)) * int(cfg.hidden_size)

    def _logits_width(self) -> int:
        """Read the local head width directly; never run a scratch forward/reset PLE."""
        head = self.model.lm_head
        if head is None or head.ndim != 2 or head.shape[0] <= 0:
            raise ValueError("a decode graph needs a two-dimensional local lm_head")
        return int(head.shape[0])

    def logits_width(self) -> int:
        return int(self.outputs.shape[-1])

    def streams(self) -> torch.Tensor | None:
        return self.streams_buf

    def _prepare(self) -> None:
        """Install request-local arena/rope metadata and independent state slots."""
        with torch.inference_mode(False):
            self.cache._decode_graph_owner = self
            self.cache.kv_width = self.width
            self.cache.rope_width = self.width
            for layer in self.cache.layers:
                if layer is not None:
                    layer.kv_width = self.width
                    if getattr(layer, "host_kv", False):
                        layer.host_position = self.start
            if self.cache.cos is None or self.cache.cos.shape[1] < self.width:
                positions = torch.arange(self.width, device=self._device).view(1, 1, -1)
                self.cache.cos, self.cache.sin = self.model.rope(
                    positions.expand(3, self.batch, -1), self.model.dtype
                )
            self.cache.share_rope_table()
            for index, (target, fields) in enumerate(self._bindings):
                frozen = self._transaction.frozen[index]
                # SlotEntries.stack owns independent positions. Reusing the frozen
                # input reference avoids retaining a second clone per list entry.
                layout = slot_layout({name: [frozen[name]] * (self.steps + 2) for name in fields})
                target.state_slots = layout
                target.state_slot = self.slot
                self._layouts.append(layout)
            self.cache.ple_history_at = lambda _ids: self.history
            self._prepare_ple_rows()
            self._set_lengths(self.start + 1)

    def _prepare_ple_rows(self) -> None:
        """Bind fixed raw-row sources while Python lookup runs during setup only."""
        tables = set()
        ple_states = getattr(self.cache, "ple_states", {})
        for index, layer in enumerate(getattr(self.model, "layers", [])):
            ple = getattr(layer, "ple", None) if layer is not None else None
            if ple is None:
                continue
            if index not in ple_states:
                raise ValueError("graph PLE layers require request-owned ple_states entries")
            table = ple.ngram_table
            required = ("_host_rows", "_gather_into", "_to_device", "head_dim", "_gather_dtype")
            if any(not hasattr(table, name) for name in required):
                raise ValueError("the PLE table does not support fixed pinned-row graph staging")
            if id(table) in tables:
                raise ValueError("graph PLE layers need distinct host-table scratch owners")
            tables.add(id(table))
            ids = ple.hasher.row_ids(self.history, 1)
            if ids.ndim != 3 or ids.shape[:2] != (self.batch, 1) or ids.shape[-1] == 0:
                raise ValueError("the PLE hasher returned an unsupported decode row layout")
            with torch.inference_mode(False):
                buffer = torch.empty(
                    ids.numel(), int(table.head_dim), dtype=table._gather_dtype,
                    device="cpu", pin_memory=self._device.type == "cuda",
                )
            table._gather_into(ids.reshape(-1).to("cpu", dtype=torch.long), buffer)
            self._ple_rows.append({"ple": ple, "table": table, "buffer": buffer, "shape": ids.shape})
        self._table_attributes = _Attributes([
            (binding["table"], ("_rows_buffer",)) for binding in self._ple_rows
        ])
        for binding in self._ple_rows:
            binding["table"]._rows_buffer = binding["buffer"]

    def _restore_table_bindings(self) -> None:
        if self._table_attributes is not None:
            self._table_attributes.restore()
            # Later eager work may reallocate a shared table's scratch buffer. Once
            # setup ends, release must not restore this old snapshot over that work.
            self._table_attributes = None

    def _refresh_ple_rows(self) -> None:
        for binding in self._ple_rows:
            ids = binding["ple"].hasher.row_ids(self.history, 1)
            if ids.shape != binding["shape"]:
                raise ValueError("PLE decode row shape changed after capture")
            binding["table"]._gather_into(
                ids.reshape(-1).to("cpu", dtype=torch.long), binding["buffer"]
            )

    def _set_lengths(self, length: int) -> None:
        self.cache.length = int(length)
        for layer in self.cache.layers:
            if layer is not None and hasattr(layer, "length"):
                layer.length = int(length)

    def _step_matrix(self, value: torch.Tensor, width: int) -> torch.Tensor:
        if value.ndim == 3 and value.shape[1] == 1:
            value = value[:, 0]
        if value.shape != (self.batch, width):
            raise ValueError(f"captured output must be [{self.batch}, {width}], got {tuple(value.shape)}")
        return value

    def _run_step(self) -> None:
        if self.streams_buf is not None:
            logits, streams = self.model.forward(
                self.inputs, cache=self.cache, past_len=self.position,
                last_token_only=True, return_streams=True,
            )
            self.streams_buf.copy_(self._step_matrix(streams, self._streams_width()).float())
        else:
            logits = self.model.forward(
                self.inputs, cache=self.cache, past_len=self.position, last_token_only=True
            )
        self.outputs.copy_(self._step_matrix(logits, self.logits_width()).float())

    def _capture(self) -> None:
        # Both passes read slot zero and write slot one. Neither is an accepted
        # decode step, regardless of placeholder tokens or host history bookkeeping.
        self._synchronize()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.inference_mode():
            self._run_step()
        torch.cuda.current_stream().wait_stream(stream)
        self._synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.inference_mode():
            self._run_step()

    def fill_history(self, seen: list[int] | list[list[int]]) -> None:
        """Stage the consumed-token tail, optionally row by row for distinct histories.

        A flat list intentionally replicates one history across the batch, preserving
        the existing replicated-prompt API. Nested lists stage one history per row.
        """
        if self._restore_error is not None:
            raise GraphRollbackError("the graph's previous rollback failed; eager fallback is unsafe")
        if self._released:
            raise GraphLifecycleError("cannot stage history for a released decode graph")
        self._history_ready = False
        try:
            if seen and isinstance(seen[0], (list, tuple)):
                if len(seen) != self.batch:
                    raise ValueError("one PLE history is required per batch row")
                rows = seen
            else:
                rows = [seen] * self.batch
            # Pinned H2D source memory must not be overwritten while a previous replay
            # may still read it. A normal token emission already drains this work.
            self._synchronize()
            self.history.fill_(int(self.model.config.primary_eos_token_id))
            for row, ids in enumerate(rows):
                tail = ids[-self._ngram:]
                if tail:
                    self.history[row, self._ngram - len(tail):] = torch.tensor(tail, dtype=torch.long)
            self._refresh_ple_rows()
            self._history_ready = True
        except BaseException:
            self.release()
            raise

    def _should_handback(self) -> bool:
        # Preserve the existing local-shard policy. It is not a global-vocabulary
        # or numerical-parity oracle, and the default zero disables it entirely.
        margin = _return_margin()
        if margin <= 0:
            return False
        local = self.outputs.reshape(-1).float()
        top = torch.topk(local, 2)
        unsure = torch.tensor(
            [1.0 if float(top.values[0] - top.values[1]) < margin * abs(float(top.values[0]))
             else 0.0], device=local.device,
        )
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(unsure, op=torch.distributed.ReduceOp.MAX)
        return float(unsure) > 0.5

    def _seed_rebased_state(self) -> None:
        # The accepted source slot remains untouched until all destination copies
        # complete. If a copy fails, release can still restore that old source.
        with torch.inference_mode():
            for layout, (_target, fields) in zip(self._layouts, self._bindings):
                for name in fields:
                    layout[name].stack[0].copy_(layout[name].stack[self._committed_slot])

    def _rebase_window(self, position: int) -> bool:
        """Seed slot zero at an exact accepted boundary without a second capture."""
        if (not self.reuse_windows or self._released
                or position != self.start + self.steps
                or self._committed_length != position or self._committed_slot != self.steps
                or position >= self._replay_limit):
            return False
        import time
        started = time.perf_counter()
        try:
            self._synchronize()
            self._seed_rebased_state()
            self._synchronize()
            self.start = int(position)
            self._committed_slot = 0
            self.rebase_count += 1
            self.rebase_s += time.perf_counter() - started
            return True
        except BaseException:
            self.release()
            raise

    def replay(self, position: int, token: torch.Tensor) -> torch.Tensor | None:
        """Replay one position and commit only after the recording chooses a token.

        A margin handback or exhausted span releases the last accepted state and
        returns None. The caller can then run eager at its current committed length;
        the failed replay's output slot and logical history are not materialized.
        """
        if self._restore_error is not None:
            raise GraphRollbackError("the graph's previous rollback failed; eager fallback is unsafe")
        if self._released:
            return None
        if position >= self._replay_limit:
            self.release()
            return None
        if not (self.start <= position < self.start + self.steps):
            if not self._rebase_window(position):
                self.release()
                return None
        try:
            self._validate_token(token)
            if self._ple_rows and not self._history_ready:
                raise ValueError("PLE replay requires fill_history before each recorded step")
            self._host_token.copy_(token)
            if not torch.equal(self.history[:, -1:], self._host_token):
                raise ValueError("staged PLE history must end in the token this replay consumes")
            self.inputs.copy_(self._host_token)
            for layer in self.cache.layers:
                if layer is not None and getattr(layer, "host_kv", False):
                    layer.host_position = int(position)
            with torch.inference_mode(False):
                self.position.fill_(position)
                self.slot.fill_(position - self.start)
            self._set_lengths(position + 1)
            self.graph.replay()
            if self._should_handback():
                self.release()
                return None
            answer = (self.outputs.argmax(-1, keepdim=True)
                      if self.select is None else self.select(self.outputs))
            self._validate_token(answer)
            self._committed_slot = (position - self.start) + 1
            self._committed_any = True
            self._committed_length = position + 1
            self._committed_history = self.history[:, 1:].clone()
            self._history_ready = False
            self._set_lengths(self._committed_length)
            for target, _fields in self._bindings:
                if hasattr(target, "initialized"):
                    target.initialized = True
            if self._transaction.attributes.value(self.cache, "ple_token_history") is not _MISSING:
                self.cache.ple_token_history = self._committed_history
            return answer
        except BaseException:
            if not self._released:
                self.release()
            raise

    def __len__(self) -> int:
        return self.steps

    def release(self) -> None:
        """Materialize the last committed slot into original buffers, exactly once."""
        if self._restore_error is not None:
            raise GraphRollbackError("the graph's previous rollback failed; eager fallback is unsafe")
        if self._released:
            return
        transaction = self._transaction
        try:
            self._synchronize()
            if transaction is not None:
                values = None
                if self._committed_any:
                    values = [
                        {name: layout[name].stack[self._committed_slot] for name in fields}
                        for layout, (_target, fields) in zip(self._layouts, self._bindings)
                    ]
                transaction.restore(
                    values=values,
                    length=self._committed_length if self._committed_any else None,
                    history=self._committed_history, committed=self._committed_any,
                )
            self._restore_table_bindings()
            self._synchronize()
            self.graph = None
            self._layouts.clear()
            self._ple_rows.clear()
            self._transaction = None
            # Local slot views/backups must not keep their storage alive while the
            # allocator cache is trimmed for the next recording.
            values = None
            transaction = None
            if self._device.type == "cuda":
                torch.cuda.empty_cache()
            self._released = True
        except BaseException as exc:
            # Best-effort Python metadata cleanup is still important on a poisoned
            # device, but it cannot certify restored GPU contents. Never report None.
            metadata_error = None
            try:
                if transaction is not None:
                    transaction.attributes.restore()
                self._restore_table_bindings()
            except BaseException as cleanup_error:
                metadata_error = cleanup_error
            self.graph = None
            self._layouts.clear()
            self._ple_rows.clear()
            self._transaction = None
            self._restore_error = metadata_error or exc
            self._released = True
            raise GraphRollbackError(
                "decode graph cache/device rollback failed; eager fallback is unsafe"
            ) from self._restore_error


def _report_capture_failure(exc: Exception) -> None:
    import traceback

    detail = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    try:
        with open("/tmp/qwen4exp_graph_window_error.txt", "a") as handle:
            handle.write(detail + "\n---\n")
    except OSError:
        pass
    print("[graph_window] capture failed: " + detail.splitlines()[0], flush=True)


def build(*, model, cache, start: int, steps: int, token: torch.Tensor, select=None,
          wants_streams: bool = False):
    """Capture one step, or decline before mutation/after a successful rollback.

    PP and non-head stages are refused before any allocation or cache mutation.
    Recoverable construction failures leave a valid eager cache; restoration or
    ownership failures propagate rather than masquerading as a safe eager fallback.
    """
    if (getattr(model, "pp_size", 1) > 1 or getattr(model, "owns_head", True) is False
            or getattr(model, "lm_head", None) is None):
        return None
    if steps <= 0 or not torch.cuda.is_available():
        return None
    # Retain this conservative validation boundary for the device oracle. Fixed-shape,
    # data-dependent top-k can be captured; this guard is not an impossibility theorem.
    # Passing it is not proof of parity either. The override enables controlled tests
    # until the sparse, past-budget path has been validated on the target device.
    budget = getattr(getattr(model, "config", None), "indexer_budget", None)
    if (budget is not None and start + steps > int(budget)
            and os.environ.get("POCKETLLM_QWEN4_GRAPH_PAST_BUDGET") != "1"):
        raise RuntimeError(
            f"--graph-window reaches {start + steps}, past the indexer budget {int(budget)}: "
            "this remains outside the current graph validation boundary. Run with a shorter "
            "span or without --graph-window; use POCKETLLM_QWEN4_GRAPH_PAST_BUDGET=1 only for "
            "controlled device-side validation. Data-dependent top-k is not inherently uncapturable."
        )
    try:
        return DecodeGraph(model=model, cache=cache, start=start, steps=steps, token=token,
                           select=select, wants_streams=wants_streams)
    except (GraphLifecycleError, GraphRollbackError):
        raise
    except Exception as exc:  # Safe eager fallback only after the constructor rolled back.
        _report_capture_failure(exc)
        return None
