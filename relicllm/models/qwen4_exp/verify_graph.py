"""A captured MTP verify round: one trunk forward over the emitted token and `K` drafts.

`graph_window.DecodeGraph` records a single position and cannot be asked for more: its state slots,
its PLE staging and its output buffer are all one row. A verify round is a different recording, not
a wider one -- it runs `K + 1` rows once, keeps only a prefix of them, and must put every rolling
state back to the end of that prefix -- so it is its own class rather than a mode of the other.

What makes the round recordable:

* **Positions are a device tensor** and every KV read uses the fixed arena width, so one recording
  replays at any offset. Both are the decode graph's existing contracts (`_write_kv`, the indexer's
  tensor branch, `_rope_cache`); nothing new is asked of attention.
* **The expert bank has a K-row kernel** (`nvfp4_moe_rows_forward`), so the MoE is one launch pair a
  layer at any row count, with no host read.
* **The rollback reads states the recording itself produced.** The gated-delta scan logs the state
  after each position, and the PLE short convolution logs the window it convolved. Both logs are
  lists the capture fills with graph-owned tensors; every replay rewrites those tensors in place,
  so after a replay the log holds *this* round's per-position states, and a partial accept is a
  copy, not a second forward.
* **The PLE rows are staged on the host.** The n-gram hash and the host-table gather are CPU work a
  recording cannot contain; the recording holds only the pinned-buffer upload, and `replay`
  refreshes the buffer from the staged history before each round -- the decode graph's scheme with
  `K + 1` rows instead of one.

The vocabulary gather and the greedy pick are recorded too, into a graph-owned `answer`, so the
draft head's recording (`DraftGraph`) can read the trunk's answers by address. The gather is
`all_gather_into_tensor` laid back out rank-major, which is the column order `gather_logits`'s
`torch.cat(parts, dim=-1)` produces, so the argmax -- first maximum, lowest global index -- is the
eager verify's.
"""

from __future__ import annotations

import os
import time

import torch

from relicllm.models.qwen4_exp.model import restore_cache, snapshot_cache


def _put_binding(target, name, value) -> None:
    if value is _MISSING:
        if name in getattr(target, "__dict__", {}):
            delattr(target, name)
    else:
        setattr(target, name, value)


def _suspend_recording_bindings(owner) -> None:
    """Temporarily undo installed Python bindings, retaining every recording-owned allocation."""
    if getattr(owner, "_bindings_retired", False) or getattr(owner, "_released", False):
        raise RuntimeError("a retired recording cannot suspend its request bindings")
    if getattr(owner, "_bindings_suspended", False):
        return
    active = []
    seen = set()
    for target, name, _old in getattr(owner, "_saved_attrs", []):
        key = (id(target), name)
        if key not in seen:
            seen.add(key)
            active.append((target, name, getattr(target, "__dict__", {}).get(name, _MISSING)))
    owner._active_request_attrs = active
    owner._active_rows_buffers = [(table, table._rows_buffer)
                                  for table, _old in getattr(owner, "_saved_rows_buffers", [])]
    for table, old in reversed(getattr(owner, "_saved_rows_buffers", [])):
        table._rows_buffer = old
    for target, name, old in reversed(getattr(owner, "_saved_attrs", [])):
        _put_binding(target, name, old)
    owner._bindings_suspended = True


def _resume_recording_bindings(owner) -> None:
    if getattr(owner, "_bindings_retired", False) or getattr(owner, "_released", False):
        raise RuntimeError("a retired recording cannot resume its request bindings")
    if not getattr(owner, "_bindings_suspended", False):
        return
    for target, name, active in owner._active_request_attrs:
        _put_binding(target, name, active)
    for table, active in owner._active_rows_buffers:
        table._rows_buffer = active
    owner._bindings_suspended = False


def _drop_suspended_bindings(owner) -> bool:
    if not getattr(owner, "_bindings_suspended", False):
        return False
    # The owner was already undone in stack order. Restoring it a second time could resurrect
    # another recording's staging after the whole session has returned to ordinary prefill.
    owner._bindings_retired = True
    owner._bindings_suspended = False
    owner._saved_attrs = []
    owner._saved_rows_buffers = []
    owner._active_request_attrs = []
    owner._active_rows_buffers = []
    return True


class VerifyGraph:
    """Record one `rows`-wide trunk forward and replay it once a round.

    The caller owns the round: it stages the feed, replays, reads the answer rows, decides how many
    positions to keep, then calls `rewind(kept)`. A one-row recording is also used for the exact
    native-boundary tail; its full forward is committed and needs no recurrent rollback. Between
    replays the cache's rolling state is the
    accepted state, so nothing else needs to know a recording exists -- except that eager trunk
    forwards on this cache are not supported while the graph is live (`ple_history_at` is staged).
    """

    def __init__(self, *, model, ctx, cache, start: int, rows: int) -> None:
        if rows < 1:
            raise ValueError("a trunk graph needs at least one emitted token")
        if getattr(model, "pp_size", 1) > 1 or getattr(model, "lm_head", None) is None:
            raise ValueError("verify graphs need a non-pipeline model with a local head")
        if int(getattr(cache, "length", -1)) != int(start):
            raise ValueError("graph start must equal the cache's committed length")
        if getattr(cache, "_decode_graph_owner", None) is not None:
            raise RuntimeError("release the cache's decode graph before recording a verify round")
        self.model = model
        self.ctx = ctx
        self.cache = cache
        self.rows = int(rows)
        self.start = int(start)
        self.device = ctx.device
        self.ngram = int(model.config.ngram_size)
        self.graph: torch.cuda.CUDAGraph | None = None
        self._gdn_logs: list = []
        self._ple_logs: list = []
        self._logs_open = False
        self._ple_rows: list = []
        self._stager: _BatchedPLEStager | None = None
        self._saved_rows_buffers: list = []
        self._saved_attrs: list = []
        self._bindings_retired = False
        self._released = False
        self.replays = 0
        #: Host seconds spent staging PLE rows, and in the replay up to its answer read.
        self.stage_s = 0.0
        self.device_s = 0.0

        self.width = self._arena_width()
        self._last_position = self.width - self.rows
        if self.start > self._last_position:
            raise ValueError("the verify span does not fit the caller's KV arena")

        before = snapshot_cache(cache)
        try:
            self._prepare()
            self._capture()
        except BaseException:
            self.release()
            restore_cache(before, cache)
            self._set_lengths(self.start)
            raise
        # Warmup and capture both ran a forward at `start` over placeholder ids. The KV and indexer
        # rows they wrote are addressed by position and overwritten by the first real round before
        # anything reads them; the rolling states are not, so they come back from the snapshot.
        restore_cache(before, cache)
        self._set_lengths(self.start)
        torch.cuda.synchronize(self.device)

    # -- setup ------------------------------------------------------------------------------------

    def _arena_width(self) -> int:
        widths = []
        for layer in self.cache.layers:
            if layer is None:
                continue
            for name in ("k", "v", "index_k"):
                tensor = getattr(layer, name, None)
                if tensor is not None:
                    widths.append(int(tensor.shape[2] if tensor.ndim == 4 else tensor.shape[1]))
        if not widths:
            raise ValueError("the cache has no KV arena to replay over")
        return min(widths)

    def _remember(self, target, name: str) -> None:
        namespace = getattr(target, "__dict__", {})
        self._saved_attrs.append((target, name, namespace.get(name, _MISSING)))

    def _prepare(self) -> None:
        cache, model = self.cache, self.model
        with torch.inference_mode(False):
            for target, name in [(cache, "kv_width"), (cache, "rope_width"),
                                 (cache, "ple_history_at")]:
                self._remember(target, name)
            cache.kv_width = self.width
            cache.rope_width = self.width
            for layer in cache.layers:
                if layer is not None:
                    self._remember(layer, "kv_width")
                    layer.kv_width = self.width
            if cache.cos is None or cache.cos.shape[1] < self.width:
                positions = torch.arange(self.width, device=self.device).view(1, 1, -1)
                cache.cos, cache.sin = model.rope(positions.expand(3, 1, -1), model.dtype)
            cache.share_rope_table()

            pinned = self.device.type == "cuda"
            self.ids = torch.zeros(1, self.rows, dtype=torch.long, device=self.device)
            self.position = torch.full((), self.start, dtype=torch.long, device=self.device)
            # The PLE window for `rows` positions: the `ngram - 1` committed tokens before the
            # first, then the feed itself.
            self.history = torch.full(
                (1, self.ngram - 1 + self.rows), int(model.config.primary_eos_token_id),
                dtype=torch.long, pin_memory=pinned,
            )
            # Both views keep the same pinned allocation alive. Host edits use its NumPy view,
            # while the graph's H2D copy reads the tensor view at a stable address.
            self._history_np = self.history.numpy()
            self._host_ids = self.history[:, self.ngram - 1:]
            committed = cache.ple_token_history
            if isinstance(committed, torch.Tensor):
                self.history[:, : self.ngram - 1].copy_(committed[:, -(self.ngram - 1):])
            cache.ple_history_at = lambda _ids: self.history

            for index, layer in enumerate(getattr(model, "layers", [])):
                ple = getattr(layer, "ple", None) if layer is not None else None
                if ple is None:
                    continue
                table = ple.ngram_table
                ids = ple.hasher.row_ids(self.history, self.rows)
                buffer = torch.empty(
                    ids.numel(), int(table.head_dim), dtype=table._gather_dtype,
                    device="cpu", pin_memory=pinned,
                )
                table._gather_into(ids.reshape(-1).to("cpu", dtype=torch.long), buffer)
                self._saved_rows_buffers.append((table, table._rows_buffer))
                table._rows_buffer = buffer
                self._ple_rows.append((ple, table, buffer, tuple(ids.shape)))
            if self._ple_rows:
                self._stager = _BatchedPLEStager(self._ple_rows, self.rows)
                # Compare both paths during setup, not in the measured decode loop. Runtime
                # staging uses this same plan and integer hash for each subsequent history.
                self._stager.check(self.history)

    def _begin_logs(self) -> None:
        self._logs_open = True
        for layer in self.cache.layers:
            if layer is not None and hasattr(layer, "state_log"):
                layer.state_log = []
        for state in (getattr(self.cache, "ple_states", None) or {}).values():
            state.state_log = []

    def _end_logs(self) -> None:
        for layer in self.cache.layers:
            if layer is not None and hasattr(layer, "state_log"):
                layer.state_log = None
        for state in (getattr(self.cache, "ple_states", None) or {}).values():
            state.state_log = None
        self._logs_open = False

    def _run(self) -> None:
        logits, streams = self.model.forward(
            self.ids, cache=self.cache, past_len=self.position, return_streams=True
        )
        if not hasattr(self, "outputs"):
            # Sized from the warmup's own answer, which runs before the capture: the recording then
            # writes into buffers that already exist and keeps the model's dtypes, so a replayed
            # round hands the head and the argmax exactly what the eager verify would.
            with torch.inference_mode(False):
                self.outputs = torch.empty_like(logits[0])
                self.streams = torch.empty_like(streams[0])
                self.answer = torch.zeros(self.rows, dtype=torch.long, device=self.device)
                self._gathered = _gather_buffer(self.outputs, self.ctx)
        self.outputs.copy_(logits[0])
        self.streams.copy_(streams[0])
        self.answer.copy_(_greedy_rows(self.outputs, self.ctx, self._gathered))

    def _capture(self) -> None:
        torch.cuda.synchronize(self.device)
        side = torch.cuda.Stream(device=self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side), torch.inference_mode():
            self._begin_logs()
            self._run()
            self._end_logs()
        torch.cuda.current_stream(self.device).wait_stream(side)
        torch.cuda.synchronize(self.device)
        # The recording allocates from a private pool, which cannot borrow the blocks the eager
        # allocator is still holding; at 256K those cached blocks are most of what is free.
        torch.cuda.empty_cache()

        # The logs are opened again for the recording, so the tensors they collect are the
        # recording's own outputs; they are taken out of the caches straight afterwards.
        self._begin_logs()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.inference_mode():
            self._run()
        if self.rows == 1:
            # A one-row native-boundary tail commits its entire forward. Its rolling buffers
            # already hold the correct state, so no per-position states or empty rollback graph
            # are needed. `rewind(..., kept=1)` updates only the logical lengths/PLE history.
            self._end_logs()
            self._rewinds = {}
            return
        for layer in self.cache.layers:
            log = getattr(layer, "state_log", None) if layer is not None else None
            if log:
                if len(log) != self.rows:
                    raise RuntimeError(
                        f"a gated-delta layer logged {len(log)} positions for {self.rows} rows"
                    )
                self._gdn_logs.append((layer, list(log)))
        for state in (getattr(self.cache, "ple_states", None) or {}).values():
            log = getattr(state, "state_log", None)
            if log:
                self._ple_logs.append((state, log[-1]))
        self._end_logs()
        if not self._gdn_logs:
            raise RuntimeError("the recording logged no gated-delta states to rewind with")
        # The rollback is ~80 small copies, and issued from Python they cost more than four
        # milliseconds a round -- launch overhead, not bytes. Each possible `kept` is its own tiny
        # recording over the same graph-owned logs, so a rewind is one launch.
        self._rewinds: dict[int, torch.cuda.CUDAGraph] = {}
        for kept in range(1, self.rows + 1):
            recording = torch.cuda.CUDAGraph()
            with torch.cuda.graph(recording), torch.inference_mode():
                self._rewind_states(kept)
            self._rewinds[kept] = recording


    # -- a round ----------------------------------------------------------------------------------

    def _set_lengths(self, length: int) -> None:
        self.cache.length = int(length)
        for layer in self.cache.layers:
            if layer is not None and hasattr(layer, "length"):
                layer.length = int(length)

    def replay(self, position: int, feed: list[int]) -> list[int]:
        """Run the round at `position` over `feed`; return the trunk's greedy answer per row.

        Row `i` answers for position `position + i + 1`. The committed PLE prefix is read from the
        cache, so it must be the one `rewind` left after the previous round.
        """
        if getattr(self, "_bindings_suspended", False):
            raise RuntimeError("the recording's request bindings are temporarily suspended")
        if getattr(self, "_bindings_retired", False):
            raise RuntimeError(
                "the verify graph's cache bindings were retired; it cannot replay over a cache "
                "that no longer points at its pinned staging"
            )
        if self.graph is None:
            raise RuntimeError("the verify graph has been released")
        if len(feed) != self.rows:
            raise ValueError(f"a recorded round consumes exactly {self.rows} ids, got {len(feed)}")
        if not (0 <= position <= self._last_position):
            raise ValueError(f"position {position} is outside the recorded arena")
        if int(self.cache.length) != int(position):
            raise ValueError("replay position must equal the cache's committed length")
        started = time.perf_counter()
        committed = self.cache.ple_token_history
        self._history_np[0, self.ngram - 1:] = feed
        if isinstance(committed, torch.Tensor):
            if committed.device.type != "cpu":
                raise ValueError("verify graph PLE history must remain on the CPU")
            self._history_np[:, :self.ngram - 1] = committed.numpy()[:, -(self.ngram - 1):]
        if self._stager is not None:
            self._stager.stage(self.history)
        self.ids.copy_(self._host_ids, non_blocking=True)
        self.position.fill_(int(position))
        staged = time.perf_counter()
        self.graph.replay()
        self.replays += 1
        # One host read a round: the answers decide what is kept, and also mark that the replay's
        # reads of the pinned staging buffers are complete before the next round rewrites them.
        answer = self.answer.tolist()
        done = time.perf_counter()
        self.stage_s += staged - started
        self.device_s += done - staged
        return answer

    def rewind(self, position: int, kept: int, feed: list[int]) -> None:
        """Leave the cache as if only `feed[:kept]` had been consumed at `position`."""
        if getattr(self, "_bindings_suspended", False):
            raise RuntimeError("the recording's request bindings are temporarily suspended")
        if getattr(self, "_bindings_retired", False):
            raise RuntimeError(
                "the verify graph's cache bindings were retired; it cannot rewind a cache it no "
                "longer owns"
            )
        if not (1 <= kept <= self.rows):
            raise ValueError(f"kept must be in [1, {self.rows}], got {kept}")
        recording = getattr(self, "_rewinds", {}).get(kept)
        if recording is not None:
            recording.replay()
        else:
            with torch.inference_mode():
                self._rewind_states(kept)
        for layer, _log in self._gdn_logs:
            layer.initialized = True
        self._set_lengths(position + kept)
        committed = self.cache.ple_token_history
        if isinstance(committed, torch.Tensor):
            if committed.device.type != "cpu" or committed.shape != (1, self.ngram - 1):
                raise ValueError("verify graph PLE history must be a CPU n-gram prefix")
            # `history` is [old prefix, full feed]. Appending only the kept feed leaves the last
            # ngram-1 entries at this fixed slice. No tensor construction/cat/clone is needed.
            committed.numpy()[:] = self._history_np[:, kept:kept + self.ngram - 1]

    def _rewind_states(self, kept: int) -> None:
        for layer, log in self._gdn_logs:
            window, recurrent = log[kept - 1]
            layer.conv_state.copy_(window.to(layer.conv_state.dtype))
            layer.recurrent_state.copy_(recurrent.to(layer.recurrent_state.dtype))
        for state, window in self._ple_logs:
            width = int(state.conv_state.shape[-1])
            state.conv_state.copy_(window[..., kept: kept + width].to(state.conv_state.dtype))

    # -- teardown ---------------------------------------------------------------------------------

    def suspend_bindings(self) -> None:
        """Undo active bindings for eager prefill while retaining graph-owned storage."""
        _suspend_recording_bindings(self)

    def resume_bindings(self) -> None:
        """Reattach stable request bindings after eager prefill on the same arena."""
        _resume_recording_bindings(self)

    def retire_bindings(self) -> None:
        """Put back the cache/PLE/table bindings this recording installed, and nothing else.

        A native-boundary handoff releases the K-row verifier and the draft chain while the one-row
        tail recording keeps replaying over the same rolling-state buffers. Destroying those two
        recordings there would free their graph pools, rewind log tensors and pinned staging
        buffers while the tail (and any in-flight replay) still addresses them, and the
        `cuda.empty_cache()` a full `release` performs costs more than the boundary step it follows.
        Retiring the *bindings* is the part the eager trunk decode needs: the cache stops pointing
        at this recording's pinned history and private table buffers, and the attributes the
        recording overwrote go back to their previous values.

        Everything that keeps the recording itself valid is retained: `graph`, `_rewinds`,
        `_gdn_logs`/`_ple_logs` (which own the per-position state tensors), `_ple_rows` (which owns
        the private pinned buffers) and `_stager`. Retirement is idempotent, performs no CUDA work
        and no deallocation, and makes `replay`/`rewind` refuse rather than fail later inside a
        recording built over a cache that no longer points at it.
        """
        if _drop_suspended_bindings(self):
            return
        if getattr(self, "_bindings_retired", False):
            return
        self._bindings_retired = True
        if getattr(self, "_logs_open", False):
            self._end_logs()
        for table, buffer in getattr(self, "_saved_rows_buffers", []):
            table._rows_buffer = buffer
        self._saved_rows_buffers = []
        # Reverse order: a target remembered twice must end with the value it held before the
        # first remember, not the intermediate installation.
        for target, name, value in reversed(getattr(self, "_saved_attrs", [])):
            if value is _MISSING:
                if name in getattr(target, "__dict__", {}):
                    delattr(target, name)
            else:
                setattr(target, name, value)
        self._saved_attrs = []

    @property
    def released(self) -> bool:
        """Whether `release` has dropped this recording's graph and buffers."""
        return getattr(self, "_released", False)

    def release(self) -> None:
        """Retire the bindings and drop the recording and every buffer it owns."""
        self.retire_bindings()
        self.graph = None
        self._rewinds = {}
        self._gdn_logs.clear()
        self._ple_logs.clear()
        self._ple_rows.clear()
        self._stager = None
        self._released = True
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


class _HeadExpertShard:
    """The head's owned FP8 experts in the shape `NvFp4SlotExpertMoE` quantizes from."""

    def __init__(self, gate_up, down, world_size: int, rank: int) -> None:
        self._layer = (gate_up, down)
        self.world_size = int(world_size)
        self.rank = int(rank)

    def layer_tensors(self, layer_idx: int):
        if layer_idx != 0:
            raise KeyError(layer_idx)
        return self._layer


def _quantize_head_experts(sharded, device) -> None:
    """Swap the head's FP8 bank for an NVFP4 one over this rank's experts only.

    The head's `HostExpertMoE` pins all 512 experts on every rank -- its reader has no expert shard,
    so `pin_layers` cannot filter by ownership -- and `ShardedMoE` then zeroes the 448 it does not
    own on every call. Its one-row path also widens each routed expert to bf16 before a bmm, which
    the round trace prices at about 3 ms of the draft round. The NVFP4 bank holds the 64 owned
    experts in the trunk's format and runs the trunk's K-row kernel, so the catch-up is one launch
    pair and the 2.5 GiB of FP8 codes go back to the allocator. It changes the head's arithmetic,
    which only the draft acceptance can see: the verify decides every emitted token.
    """
    from relicllm.models.qwen4_exp.moe import NvFp4SlotExpertMoE
    from relicllm.models.qwen4_exp.quant import FP8Tensor

    inner = sharded.inner
    if isinstance(inner, NvFp4SlotExpertMoE):
        return
    world, rank = int(sharded.world_size), int(sharded.rank)
    num_experts = int(inner.num_experts)
    owned = list(range(rank, num_experts, world))
    with torch.inference_mode(False):
        stacked = inner._fp8_stacks.get(0)
        if stacked is not None:
            order = inner._fp8_order[0]
            where = {expert: slot for slot, expert in enumerate(order)}
            pick = torch.tensor([where[e] for e in owned], dtype=torch.long, device=device)
            gate_code, gate_scale, down_code, down_scale = (t.index_select(0, pick) for t in stacked)
            block = inner._fp8_blocks[0]
        else:
            bank = inner._resident.get(0) or {}
            if not all(e in bank for e in owned):
                raise RuntimeError("the head's experts are not resident; cannot quantize them")
            gate_code = torch.stack([bank[e][0].code for e in owned])
            gate_scale = torch.stack([bank[e][0].scale for e in owned])
            down_code = torch.stack([bank[e][1].code for e in owned])
            down_scale = torch.stack([bank[e][1].scale for e in owned])
            block = bank[owned[0]][0].block
        shard = _HeadExpertShard(
            FP8Tensor(gate_code, gate_scale, block), FP8Tensor(down_code, down_scale, block),
            world, rank,
        )
        backend = NvFp4SlotExpertMoE(shard, num_experts, device, num_layers=1)
        # The backend retains its shard object, so deleting only the local `shard` name would
        # keep the original FP8 tensors alive beside the quantized table for every request.
        shard._layer = None
    sharded.inner = backend
    del shard, gate_code, gate_scale, down_code, down_scale
    inner._fp8_stacks.pop(0, None)
    inner._fp8_lookup.pop(0, None)
    inner._resident.pop(0, None)
    if device.type == "cuda":
        torch.cuda.empty_cache()


class _BatchedPLEStager:
    """Hash and gather every PLE layer's rows for a round in one pass.

    Per layer, `NGramHasher.row_ids` builds its shifted copies of the history and mixes them with
    that layer's multipliers, and `HostNGramTable._gather_into` copies the named rows one torch
    slice at a time; across 48 layers and a round's rows that is several milliseconds of
    Python-level calls for a few kilobytes. The shifted copies depend only on the history and the
    EOS id, so they are built once; the mixing is the same int64 arithmetic with the layers'
    multipliers stacked on a leading axis, so its wraparound and floor-remainder are the per-layer
    ones elementwise; and the gather writes through numpy views of the same shards and pinned
    buffers. `check` compares the result against the per-layer path.
    """

    def __init__(self, plan: list, rows: int) -> None:
        if not plan or rows < 1:
            raise ValueError("PLE staging needs a nonempty layer plan and positive row count")
        hashers = [ple.hasher for ple, _table, _buffer, _shape in plan]
        first = hashers[0]
        for hasher in hashers:
            if (hasher.ngram_size, hasher.heads_per_ngram, hasher.eos_token_id) != (
                first.ngram_size, first.heads_per_ngram, first.eos_token_id
            ):
                raise ValueError("PLE layers disagree on the n-gram layout")
        self.rows = int(rows)
        self.hasher = first
        self.multipliers = torch.stack([h.layer_multipliers.to(torch.long) for h in hashers])
        self.sizes = torch.stack([h.head_vocab_sizes.to(torch.long) for h in hashers])
        self.offsets = torch.stack([h.head_offsets.to(torch.long) for h in hashers])
        self._multipliers_np = self.multipliers.numpy()
        self._sizes_np = self.sizes.numpy()
        self._offsets_np = self.offsets.numpy()
        self.plan = plan
        self.targets = []
        for _ple, table, buffer, _shape in plan:
            shards = [_numpy_view(shard) for shard in table._gather_shards]
            self.targets.append((shards, _numpy_view(buffer), int(table.rows_per_shard)))

    def _row_ids_numpy(self, history: torch.Tensor):
        """The reference hash in NumPy int64, with identical wraparound and EOS masking.

        This is host bookkeeping on a tiny window, not tensor compute. Running the tensor version
        here invokes PyTorch's intra-op pool for the shifted copies and remainders; on the 4090
        host even a four-row window costs milliseconds. NumPy's integer ufuncs stay on the calling
        thread. Signed int64 multiplication/XOR wraps exactly as the reference, and remainder with
        a positive divisor is floor modulo in both implementations.
        """
        import numpy as np

        if history.device.type != "cpu" or history.dtype != torch.long:
            raise ValueError("PLE staging needs CPU int64 token history")
        if history.ndim != 2 or history.shape[0] != 1 or history.shape[1] < self.rows:
            raise ValueError("PLE staging needs one history with at least the output row count")
        tokens = history.numpy()[0]
        positions = np.arange(tokens.size, dtype=np.int64)
        previous_eos = np.maximum.accumulate(np.where(tokens == self.hasher.eos_token_id, positions, -1))
        previous_eos = np.concatenate((np.array([-1], dtype=np.int64), previous_eos[:-1]))
        wanted = positions[-self.rows:]
        distance = wanted - (previous_eos[-self.rows:] + 1)
        shifted = []
        for shift in range(self.hasher.ngram_size):
            source = wanted - shift
            valid = (source >= 0) & (distance >= shift)
            shifted.append(np.where(valid, tokens[source.clip(min=0)], self.hasher.eos_token_id))
        blocks = []
        for ngram in range(2, self.hasher.ngram_size + 1):
            start = (ngram - 2) * self.hasher.heads_per_ngram
            end = start + self.hasher.heads_per_ngram
            mixed = shifted[0][None, :] * self._multipliers_np[:, 0:1]
            for position in range(1, ngram):
                mixed ^= shifted[position][None, :] * self._multipliers_np[:, position:position + 1]
            blocks.append(
                np.remainder(mixed[..., None], self._sizes_np[:, None, start:end])
                + self._offsets_np[:, None, start:end]
            )
        return np.concatenate(blocks, axis=-1)

    def row_ids(self, history: torch.Tensor) -> torch.Tensor:
        """`[layers, rows, ngram_heads]`, equal to stacking each layer's `row_ids(history, rows)`."""
        return torch.from_numpy(self._row_ids_numpy(history))

    def stage(self, history: torch.Tensor) -> None:
        ids = self._row_ids_numpy(history).reshape(len(self.targets), -1).tolist()
        for (shards, buffer, per_shard), names in zip(self.targets, ids):
            limit = len(shards)
            for position, row in enumerate(names):
                shard, local = divmod(row, per_shard)
                if shard >= limit:
                    buffer[position] = 0
                else:
                    buffer[position] = shards[shard][local]

    def check(self, history: torch.Tensor) -> None:
        """Raise unless the batched hash and gather reproduce what the per-layer path staged."""
        ids = self.row_ids(history)
        for index, (ple, table, buffer, _shape) in enumerate(self.plan):
            expected = ple.hasher.row_ids(history, self.rows).reshape(-1)
            if not torch.equal(ids[index].reshape(-1), expected):
                raise RuntimeError(f"batched PLE hash differs from layer {index}'s own")
        # The reference is staged here, by the per-layer path, so the comparison does not depend on
        # what the buffers held before the call.
        staged = []
        for ple, table, buffer, _shape in self.plan:
            names = ple.hasher.row_ids(history, self.rows).reshape(-1).to("cpu", dtype=torch.long)
            table._gather_into(names, buffer)
            staged.append(buffer.clone())
        for _ple, _table, buffer, _shape in self.plan:
            buffer.zero_()
        self.stage(history)
        for index, (_ple, _table, buffer, _shape) in enumerate(self.plan):
            if not torch.equal(buffer, staged[index]):
                raise RuntimeError(f"batched PLE gather differs from layer {index}'s own")


def _numpy_view(tensor: torch.Tensor):
    """A numpy view of a CPU tensor's bytes, by integer reinterpretation for bf16/fp8."""
    if tensor.dtype in (torch.bfloat16, torch.float16):
        return tensor.view(torch.int16).numpy()
    if tensor.dtype in (torch.float8_e4m3fn, torch.uint8):
        return tensor.view(torch.uint8).numpy()
    return tensor.numpy()


def _gather_buffer(local: torch.Tensor, ctx) -> torch.Tensor | None:
    """The rank-stacked landing buffer for `_greedy_rows`, or None without tensor parallelism."""
    world = int(getattr(ctx, "tp_world_size", 1))
    if world <= 1:
        return None
    return torch.empty((world * local.shape[0],) + tuple(local.shape[1:]),
                       dtype=local.dtype, device=local.device)


def _greedy_rows(local: torch.Tensor, ctx, gathered: torch.Tensor | None) -> torch.Tensor:
    """`gather_logits(local).argmax(-1)` for `[rows, vocab_shard]` logits, recordable.

    `all_gather_into_tensor` stacks the shards rank-major along rows; transposing that back to
    `[rows, rank, shard]` and flattening the last two is exactly `torch.cat(parts, dim=-1)`.
    """
    if gathered is None:
        return local.argmax(-1)
    import torch.distributed as dist

    dist.all_gather_into_tensor(gathered, local.contiguous())
    world = gathered.shape[0] // local.shape[0]
    full = gathered.view(world, local.shape[0], -1).transpose(0, 1).reshape(local.shape[0], -1)
    return full.argmax(-1)


class DraftGraph:
    """Record the draft head's whole round: the catch-up over every verify row, then the chain.

    The eager loop calls the head `1 + (K - 1)` times a round from Python, each call a host read
    for its argmax, and at 256K those calls cost more than the trunk's recorded verify. Two facts
    let one recording do all of them with no host read:

    * **The catch-up's ids are the trunk's answers.** Convention B pairs the trunk state at row `j`
      with the token at `j + 1`; for every kept row that token is either an accepted draft, which
      equals the trunk's answer at `j`, or the bonus, which *is* the answer at `j`. So the catch-up
      over all `K + 1` rows takes `VerifyGraph.answer` as its ids, and the rows past the kept prefix
      are wasted work, not wrong work: their keys land at positions the chain or the next round
      writes before anything reads them, the same contract the eager chain's speculative keys rely
      on. What the kept prefix decides is only which row seeds the chain -- `kept - 1` -- and that is
      a device index.
    * **The head reads only position-addressed state.** It is one sparse-attention layer over its
      own KV arena, with no recurrence and no PLE term, so nothing needs rewinding.

    The expert bank goes through its one-row, device-indexed path (`HostExpertMoE._from_fp8_stack`,
    one row at a time through `ShardedMoE.rowwise`), which is the only head MoE path a recording
    can hold.
    """

    def __init__(self, *, model, ctx, head_cache, verify: VerifyGraph, drafts: int) -> None:
        if drafts < 1 or drafts + 1 != verify.rows:
            raise ValueError("a draft graph seeds exactly the verify graph's draft count")
        self.model = model
        self.head = model.mtp
        self.ctx = ctx
        self.cache = head_cache
        self.verify = verify
        self.drafts = int(drafts)
        self.rows = verify.rows
        self.device = ctx.device
        self.graph: torch.cuda.CUDAGraph | None = None
        self.replays = 0
        self._saved_attrs: list = []
        self._bindings_retired = False
        self._released = False
        layer = head_cache.layers[0]
        if getattr(layer, "cos", None) is None:
            raise ValueError("the head cache needs its absolute RoPE table (attach_rope_table)")
        self.width = int(layer.k.shape[2])
        if verify.start + 2 * self.rows > self.width:
            raise ValueError("the head arena does not fit a round past the verify start")
        with torch.inference_mode(False):
            self.position = torch.full((), verify.start, dtype=torch.long, device=self.device)
            self.select = torch.zeros((), dtype=torch.long, device=self.device)
            self.out = torch.zeros(self.drafts, dtype=torch.long, device=self.device)
        try:
            self._prepare()
            self._capture()
        except BaseException:
            self.release()
            raise
        layer.length = int(verify.start)
        torch.cuda.synchronize(self.device)

    def _prepare(self) -> None:
        layer = self.cache.layers[0]
        self._saved_attrs.append((layer, "kv_width", layer.__dict__.get("kv_width", _MISSING)))
        layer.kv_width = self.width
        # `MTPHead.moe` is the backend its decoder layer routes through (`ShardedMoE` under TP).
        self._moe = getattr(self.head, "moe", None)
        if self._moe is None or not hasattr(self._moe, "rowwise"):
            raise RuntimeError("the head's expert backend is not a ShardedMoE to run row-wise")
        if os.environ.get("POCKETLLM_QWEN4_MTP_NVFP4", "0") == "1":
            _quantize_head_experts(self._moe, self.device)
        inner = getattr(self._moe, "inner", None)
        from relicllm.models.qwen4_exp.moe import NvFp4SlotExpertMoE

        if isinstance(inner, NvFp4SlotExpertMoE):
            # The K-row kernel takes the whole catch-up in one launch pair, reading 4-bit in place.
            return
        if inner is None or not hasattr(inner, "_capture_safe"):
            raise RuntimeError("the head's expert backend has no device-indexed path")
        self._saved_attrs.append((self._moe, "rowwise", self._moe.rowwise))
        self._moe.rowwise = True
        # Left on after release: building the device stack drops the per-expert dict it replaces,
        # and the host-read path that dict served cannot run without it.
        inner._capture_safe = True

    def _token(self, mixed: torch.Tensor) -> torch.Tensor:
        local = torch.nn.functional.linear(mixed[:, -1], self.model.lm_head)
        if not hasattr(self, "_gathered"):
            with torch.inference_mode(False):
                self._gathered = _gather_buffer(local, self.ctx)
        return _greedy_rows(local, self.ctx, self._gathered).reshape(1, 1)

    def _recycle(self, mixed: torch.Tensor, stream: torch.Tensor) -> torch.Tensor:
        from relicllm.models.qwen4_exp.runtime import _mtp_recycle

        if _mtp_recycle() == "collapsed":
            return mixed.unsqueeze(-2).expand(-1, -1, self.head.config.hc_count, -1).reshape_as(stream)
        return stream

    def _run(self) -> None:
        tokens = self.verify.answer.view(1, -1)
        mixed, stream = self.head.forward(
            self.verify.streams.unsqueeze(0), tokens, self.cache,
            past_len=self.position, return_streams=True,
        )
        pick = self.select.view(1)
        mixed = mixed.index_select(1, pick)
        stream = stream.index_select(1, pick)
        token = self._token(mixed)
        self.out[0:1].copy_(token.reshape(1))
        stack = self._recycle(mixed, stream)
        base = self.position + self.select + 1
        for link in range(1, self.drafts):
            mixed, stream = self.head.forward(
                stack, token, self.cache, past_len=base + (link - 1), return_streams=True
            )
            token = self._token(mixed)
            self.out[link:link + 1].copy_(token.reshape(1))
            stack = self._recycle(mixed, stream)

    def _capture(self) -> None:
        torch.cuda.synchronize(self.device)
        side = torch.cuda.Stream(device=self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side), torch.inference_mode():
            self._run()
        torch.cuda.current_stream(self.device).wait_stream(side)
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.inference_mode():
            self._run()

    def replay(self, position: int, kept: int) -> list[int]:
        """Catch up over the verify round at `position`, seed from row `kept - 1`, chain on.

        Must follow `VerifyGraph.replay` for the same round: it reads that recording's `answer`
        and `streams`. Returns the next round's `K` drafts, the first for `position + kept + 1`.
        """
        if getattr(self, "_bindings_suspended", False):
            raise RuntimeError("the recording's request bindings are temporarily suspended")
        if getattr(self, "_bindings_retired", False):
            raise RuntimeError(
                "the draft graph's cache bindings were retired; it cannot replay over a head "
                "cache that no longer points at it"
            )
        if self.graph is None:
            raise RuntimeError("the draft graph has been released")
        if not (1 <= kept <= self.rows):
            raise ValueError(f"kept must be in [1, {self.rows}], got {kept}")
        if position + kept + self.drafts > self.width:
            raise ValueError("the chain would run past the head arena")
        self.position.fill_(int(position))
        self.select.fill_(int(kept) - 1)
        self.graph.replay()
        self.replays += 1
        return self.out.tolist()

    def suspend_bindings(self) -> None:
        """Undo active bindings for eager prefill while retaining graph-owned storage."""
        _suspend_recording_bindings(self)

    def resume_bindings(self) -> None:
        """Reattach stable request bindings after eager prefill on the same arena."""
        _resume_recording_bindings(self)

    def retire_bindings(self) -> None:
        """Put back the head cache attributes this recording installed, and nothing else.

        The draft chain is released at the native boundary while one-row tail replays still run.
        Restoring only the bindings keeps the recording, its graph pool, its `out`/`select`/`position`
        buffers and its gathered buffer alive; `release` is what frees them, once the request is
        completely finished. Idempotent, no CUDA work, no deallocation.
        """
        if _drop_suspended_bindings(self):
            return
        if getattr(self, "_bindings_retired", False):
            return
        self._bindings_retired = True
        # Reverse order: a target remembered twice must end with the value it held before the
        # first remember, not the intermediate installation.
        for target, name, value in reversed(getattr(self, "_saved_attrs", [])):
            if value is _MISSING:
                if name in getattr(target, "__dict__", {}):
                    delattr(target, name)
            else:
                setattr(target, name, value)
        self._saved_attrs = []

    @property
    def released(self) -> bool:
        """Whether `release` has dropped this recording's graph and buffers."""
        return getattr(self, "_released", False)

    def release(self) -> None:
        """Retire the bindings and drop the recording and every buffer it owns."""
        self.retire_bindings()
        self.graph = None
        self._released = True
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


_MISSING = object()


__all__ = ["DraftGraph", "VerifyGraph"]
