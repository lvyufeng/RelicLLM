# One scheduler, lifted

**Date:** 2026-10-07
**Question:** issue #130 — lift the scheduler out of `deepseek_v4` so every runtime reaches one
model-agnostic queue. What is actually there to lift, and what does width 1 mean once it is lifted?

**Answer: the queue #130 described had no callers; the thing that ran was the phase policy. So the
lift is not a move — it is a *build*:** a new `relicllm/scheduler/` holding one request-at-a-time
queue that replaces the per-adapter `_request_lock`, and a `deepseek_v4/pd_scheduler.py` reduced to
the one part that is V4's. This page records the measurements that decided it.

## What was measured

`PDScheduler` in `relicllm/models/deepseek_v4/pd_scheduler.py` declared five queue methods. Every one
of them has **zero callers** anywhere in the tree:

| symbol | callers |
|---|---|
| `PDScheduler.submit` / `has_work` / `next_step` / `mark_prefill_done` / `mark_request_done` | 0 |
| `Request` (the queued row) | 0 |
| `has_phase_overrides` / `apply_phase_resources` | 0 direct; reached only through the facade's `phase_callback` |
| `run_single_stream_request` | 0 |
| `PDExecutionFacade` / `PDExecutionConfig` | `models/deepseek_v4/{serving,generation}.py` |
| `PDPhasePolicy` (was part of the above) | the facade, and the generation loop's `phase_callback` |

The five queue methods are dead. The V4 HTTP path is `_run_payload → executor.run(...)`, one request
at a time; nothing ever called `submit`, so nothing was ever queued. The facade and the phase policy
do run.

Two consequences follow, and they are the whole design:

- **The queue had to be written, not moved.** A port that relocated the dead methods into
  `relicllm/scheduler/` would have put dead code where it reads as load-bearing. What the shared
  module holds instead is a queue that takes over the one thing every runtime already serialized on:
  `BackendBase`'s `_request_lock`.
- **The V4 phase machinery is not scheduler code.** `apply_phase_resources` pins CPU sets, OMP thread
  counts and NUMA nodes and swaps the INT8 attention variant, all from `DEEPSEEK_PD_*` — it is the
  host-resident-expert PD run's resource policy, and no other runtime in this tree has one. It stays
  in `models/deepseek_v4/pd_scheduler.py`.

## What replaced the lock, and why

Every adapter held a `threading.RLock` around its generation call. A second request waited on the
mutex; nothing about the waiting request was visible, and the mutex had no idea which request it was
for. `relicllm/scheduler/core.py` makes that lock a queue:

- **`Request` is a row of counters** — `num_computed_tokens`, `num_tokens_with_spec` — not a phase
  enum, for the reason vLLM states for its own scheduler: chunked prefill, prefix reuse and
  speculative decoding are then counters on a request rather than branches in a scheduler. `phase`
  survives as a *view* derived from the counters, so the V4 call sites that think in
  prefill/decode keep their word with one source of truth behind it.
- **`Scheduler.submit` / `acquire` / `mark_request_done`** are the queue. `acquire` is a context
  manager over one slot; the slot is the same one-slot mutex the lock was, so at width 1 the
  serialization is identical. What changed is that a request is readable while it waits.
- **The model is data, never a branch.** `ExecutionPlan` carries the chunked-prefill size and the
  phase callback as opaque values the scheduler forwards without reading. The V4 phase policy is
  one caller's plan; another model's is another plan.

`BackendBase` now owns a `Scheduler` and its `__init__` no longer creates `_request_lock`. The two
`RuntimeAdapter` engine-call sites and the two `TorchBackend` ones take the slot through
`self._scheduler.acquire(...)`; the four adapters that created their own `_request_lock` no longer
do (`torch` keeps a `_load_lock`, which is a different lock and was always doing a different job).

## Width 1, held

The acceptance is that a served request is byte-identical before and after. It holds because the
slot is taken and released at the same two places the lock was, and no behaviour reads the queue:
`submit` appends, `acquire` blocks exactly as the mutex did, the slot is released on the same exit.
The streamed paths keep the slot across the live event loop exactly as the lock was held across it —
a buffered variant was drafted and discarded, because releasing the engine early would have changed
streaming latency, which is not a "no behaviour change". The per-runtime serving tests are the pin.

One difference is admitted rather than hidden: a queue shows the idle slot as *one* waiter, so the
fairness of two contenders is arrival order rather than whatever the lock's hand-off happened to be.
At width 1 with one request at a time this is not observable in a single-request answer; it is
observable only under concurrency, which no runtime claims (`supports_batch=False` everywhere).

## What the scheduler may not do

It may not name an architecture, and it may not read scheduling policy from the environment.
`tests/test_scheduler_core.py` scans `relicllm/scheduler/**` for both: no `relicllm.models` import,
no string or identifier equal to a `--backend` runtime name, no `os.environ` read but `RANK` (which
spaces request ids, not policy). This is the shape of #130's acceptance item 4, and it is what keeps
a second model's arrival a new `ExecutionPlan` rather than a new branch.

`relicllm/scheduler/` sits in the **device** layer of `tests/test_package_boundaries.py`, beside
`runtime/`: both are model- and format-agnostic, and the scheduler is the plane every runtime reaches
the engine through.

## What this is not

- **Not continuous batching.** That is #33, and it needs #129's KV declaration to account admission
  against. At width 1 nothing batches; the scheduler is the place batching lands, and its counter
  state is the shape batching extends.
- **Not the phase policy generalized.** `PDPhasePolicy` stays V4's. If a second model grows a
  host-resident-expert run it copies the shape, or the policy is lifted then — not before, and not by
  widening the scheduler.

## Relation to the open tree

- **#130** — this page is its record. The lift is done; the width-1 acceptance is the per-runtime
  serving tests.
- **#129** — the KV declaration admission will account against next. The scheduler reads no KV numbers
  yet because there are none to read.
- **#33 / #34** — continuous batching and paged KV, both downstream of #129 and of this queue.
- **#18** — "port the C++ BatchScheduler loop to a pure-Python scheduler". This is that Python
  scheduler existing once; #18 narrows to its porting detail or closes into this.