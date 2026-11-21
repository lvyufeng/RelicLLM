"""Task-private profiling of the actual graph/tile decode, with no serving patches.

Two separate passes follow a 16-forward uninstrumented window:
(1) eight event-scoped forwards and excluded device-route telemetry;
(2) four all-rank CANN CPU/NPU trace forwards. Profiler sums are not TPS.
The start KV/hash boundary is the valid context-24 boundary, not a prefill
snapshot resumed at an unrelated position.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
import inspect
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist

import profile_actual_cube as probe


class ReplayProxy:
    def __init__(self, original, owner, label, layer):
        self.original, self.owner, self.label, self.layer = original, owner, label, layer

    def replay(self):
        return self.owner.call(self.label, self.layer, self.original.replay)

    def __getattr__(self, name):
        return getattr(self.original, name)


class ScopeRecorder:
    def __init__(self, model, *, prefetch, events, annotations, telemetry):
        self.model, self.prefetch = model, prefetch
        self.events, self.annotations, self.telemetry = events, annotations, telemetry
        self.records, self.routes, self.worker_records, self.replacements = [], [], [], []
        self.step = None

    def call(self, label, layer, function, *args, **kwargs):
        begin = torch.npu.Event(enable_timing=True) if self.events else None
        end = torch.npu.Event(enable_timing=True) if self.events else None
        if begin is not None:
            begin.record()
        start = time.perf_counter()
        with torch.profiler.record_function(f"relic.scope.{label}.layer{layer}") if self.annotations else nullcontext():
            value = function(*args, **kwargs)
        elapsed = time.perf_counter() - start
        if end is not None:
            end.record()
        self.records.append((self.step, label, layer, elapsed, begin, end))
        return value

    def replace(self, owner, name, value):
        self.replacements.append((owner, name, getattr(owner, name)))
        setattr(owner, name, value)

    def install(self):
        for block in self.model.layers:
            graphs = block.decode_graph
            if graphs is None or not graphs.ready:
                raise RuntimeError("scope profiling must use already captured current decode graphs")
            for name in ("graph_a", "graph_a_emit", "graph_a_fill", "graph_b"):
                graph = getattr(graphs, name)
                if graph is not None:
                    label = "layer_B_shared_MoE_reduce_HC" if name == "graph_b" else "layer_A_attention_router_HC"
                    self.replace(graphs, name, ReplayProxy(graph, self, label, block.layer_id))
            store = block.ffn.routed
            original = store.forward
            def routed(*args, _original=original, _block=block, **kwargs):
                result = self.call("routed_experts", _block.layer_id, _original, *args, **kwargs)
                if self.telemetry:
                    # A small D2D clone after the timed scope, not a routing D2H.
                    # All telemetry is copied back after this separate pass ends.
                    self.routes.append((self.step, _block.layer_id, args[2].detach().clone()))
                return result
            self.replace(store, "forward", routed)
            if block.engram is not None:
                original_engram = block.engram.forward
                def engram(*args, _original=original_engram, _layer=block.layer_id, **kwargs):
                    return self.call("complete_CPU_Engram_boundary", _layer, _original, *args, **kwargs)
                self.replace(block.engram, "forward", engram)
        if self.prefetch is not None and self.prefetch.enabled:
            original_compute = self.prefetch._compute
            def compute(engram, hashes):
                start = time.perf_counter()
                try:
                    with torch.profiler.record_function("relic.scope.CPU_Engram_lookup_projection_worker") if self.annotations else nullcontext():
                        return original_compute(engram, hashes)
                finally:
                    self.worker_records.append({"layer_hash_index": engram.layer_hash_index,
                                                "elapsed_ms": (time.perf_counter() - start) * 1000})
            self.replace(self.prefetch, "_compute", compute)
        return self

    def close(self):
        for owner, name, original in reversed(self.replacements):
            setattr(owner, name, original)
        self.replacements.clear()

    def result(self, steps):
        totals = defaultdict(lambda: {"calls": 0, "host_inclusive_ms": 0.0, "stream_elapsed_ms": 0.0})
        by_layer = defaultdict(lambda: {"calls": 0, "host_inclusive_ms": 0.0, "stream_elapsed_ms": 0.0})
        raw = []
        for step, label, layer, host, begin, end in self.records:
            if step is None:
                continue
            event_ms = begin.elapsed_time(end) if begin is not None else None
            for row in (totals[label], by_layer[(label, layer)]):
                row["calls"] += 1
                row["host_inclusive_ms"] += host * 1000
                if event_ms is not None:
                    row["stream_elapsed_ms"] += event_ms
            raw.append({"step": step, "scope": label, "layer": layer, "host_inclusive_ms": host * 1000,
                        "stream_elapsed_ms": event_ms})
        for row in totals.values():
            row["host_inclusive_ms_per_forward"] = row["host_inclusive_ms"] / steps
            row["stream_elapsed_ms_per_forward"] = row["stream_elapsed_ms"] / steps
            row["calls_per_forward"] = row["calls"] / steps
        rank, world = dist.get_rank(), dist.get_world_size()
        telemetry = []
        if self.routes:
            ids = torch.stack([entry[2][0] for entry in self.routes]).cpu().tolist()
            for (step, layer, _), selected in zip(self.routes, ids):
                count = sum(expert % world == rank for expert in selected)
                telemetry.append({"step": step, "layer": layer, "global_ids": selected, "owned": count})
        return {"rank": rank, "steps": steps, "totals": dict(totals), "raw_scopes": raw,
                "per_layer": [{"scope": label, "layer": layer, **row} for (label, layer), row in by_layer.items()],
                "excluded_route_telemetry": telemetry, "CPU_worker_calls": self.worker_records,
                "notes": ["host-inclusive times overlap stream times and must not be added",
                          "NPU events measure stream/dependency elapsed, not pure kernel execution",
                          "CPU Engram stream scope includes its CPU handoff and broadcast",
                          "route telemetry is a separate-pass D2D clone and post-pass D2H, never a TPS-window read",
                          "the eight event-scoped forwards are not the 16-forward TPS denominator"]}


def run_current_profile(model, candidate_forward, position, first, output, reset, *,
                        prefetch, expected_tokens, steps=4):
    import torch_npu

    rank = dist.get_rank()
    output = Path(output)
    sample_steps = 8
    reset()
    dist.barrier()
    probe.sync()
    recorder = ScopeRecorder(model, prefetch=prefetch, events=True, annotations=False, telemetry=True).install()
    prior_wait = prefetch.stats["CPU_wait_seconds"] if prefetch is not None else 0.0
    token, sample_tokens, sample_wall = first, [], []
    try:
        for step in range(sample_steps):
            recorder.step = step
            start = time.perf_counter()
            _, logits, _ = candidate_forward(torch.tensor([[token]]), position + step)
            probe.sync()
            token = int(logits[0].argmax().item())
            sample_wall.append(time.perf_counter() - start)
            sample_tokens.append(token)
        probe.sync()
        scopes = recorder.result(sample_steps)
    finally:
        recorder.close()
    scopes.update(context_start=position, context_end=position + sample_steps,
                  wall_ms_per_forward=sum(sample_wall) / sample_steps * 1000,
                  step_wall_ms=[value * 1000 for value in sample_wall], token_ids=sample_tokens,
                  same_token_ids_as_uninstrumented_window=sample_tokens == expected_tokens[:sample_steps],
                  exposed_prefetch_wait_ms_per_forward=((prefetch.stats["CPU_wait_seconds"] - prior_wait)
                                                        / sample_steps * 1000) if prefetch is not None else None)
    if not scopes["same_token_ids_as_uninstrumented_window"]:
        raise RuntimeError("event-scoped profile changed the current model's token sequence")
    with (output / f"decode-scopes-rank-{rank}.json").open("x") as stream:
        json.dump(scopes, stream, indent=2)
    reset()
    dist.barrier()
    probe.sync()
    directory = output / f"profiler-rank{rank}"
    directory.mkdir(exist_ok=False)
    recorder = ScopeRecorder(model, prefetch=prefetch, events=False, annotations=True, telemetry=False).install()
    experimental = torch_npu.profiler.experimental_config._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1)
    token, trace_tokens, trace_wall = first, [], []
    activities = [torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU]
    try:
        with torch_npu.profiler.profile(
                activities=activities, experimental_config=experimental, record_shapes=True,
                with_stack=False, on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(directory))) as profiler:
            # Every rank is profiled. Keep setup skew separate from forward markers.
            dist.barrier()
            probe.sync()
            for step in range(steps):
                recorder.step = step
                with torch.profiler.record_function(f"relic.decode_step.{step}"):
                    start = time.perf_counter()
                    _, logits, _ = candidate_forward(torch.tensor([[token]]), position + step)
                    probe.sync()
                    token = int(logits[0].argmax().item())
                    trace_wall.append(time.perf_counter() - start)
                    trace_tokens.append(token)
                profiler.step()
        probe.sync()
    finally:
        recorder.close()
    if trace_tokens != expected_tokens[:steps]:
        raise RuntimeError("CANN profiling changed the current model's token sequence")
    reset()
    result = {"rank": rank, "all_ranks_profiled": True, "profile_directory": str(directory),
              "steps": steps, "context_start": position, "context_end": position + steps,
              "token_ids": trace_tokens, "same_token_ids_as_uninstrumented_window": True,
              "profile_wall_ms_per_forward": sum(trace_wall) / steps * 1000,
              "event_scope_receipt": str(output / f"decode-scopes-rank-{rank}.json"),
              "note": "actual captured graph and routed path; kernel/dependency attribution, not a TPS window"}
    with (output / f"profile-scope-rank-{rank}.json").open("x") as stream:
        json.dump(result, stream, indent=2)
    return result


def main():
    # Patch only the task-private generation function. All model code stays the
    # current measured implementation, and every source is hashed by the runner.
    source = inspect.getsource(probe.generate_probe)
    old = "        position += 4\n" if "        position += 4\n" in source else "    position += 4\n"
    indent = old.split("position")[0]
    addition = old + (
        f"{indent}profile_cache_boundary = snapshot(model)\n"
        f"{indent}profile_hash_boundary = front.snapshot_prefix(model.max_seq_len)\n"
        f"{indent}profile_first_token = token\n"
        f"{indent}def reset_profile_boundary():\n"
        f"{indent}    restore(model, profile_cache_boundary)\n"
        f"{indent}    front.restore_prefix(profile_hash_boundary)\n"
        f"{indent}    sync()\n")
    if source.count(old) != 1:
        raise RuntimeError("the 4-warmup profiling boundary is not unique")
    source = source.replace(old, addition, 1)
    original_call = "run_kernel_profile(model, candidate_forward, position, first, OUTPUT, reset)"
    if source.count(original_call) != 1:
        raise RuntimeError("the task-private profiling hook moved")
    source = source.replace(original_call,
        "run_kernel_profile(model, candidate_forward, position, profile_first_token, OUTPUT, reset_profile_boundary, "
        "prefetch=prefetch, expected_tokens=timed_tokens)", 1)
    exec(compile(source, str(Path(__file__).resolve()), "exec"), probe.__dict__)
    probe.run_kernel_profile = run_current_profile
    return probe.main()


if __name__ == "__main__":
    raise SystemExit(main())
