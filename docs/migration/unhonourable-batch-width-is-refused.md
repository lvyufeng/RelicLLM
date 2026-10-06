# A `--max-batch-size` that cannot be honoured was refused

**Affects, historically:** anyone who passed `--max-batch-size` above 1, or `--enable-batching`, to
`pocketllm serve --backend cpp` on a build or a checkpoint that could not provide the batch
scheduler.

**Not reachable today.** The `cpp` backend and the native `BatchScheduler` were removed with the
`cpp_engine/` tree, so there is no build or checkpoint on which this note applies. It is kept
because it is the record of a refusal the code still carries: `--backend cpp` is now itself refused,
by name, with a sentence naming the retirement — see
[the native surface decision](../architecture/native_surface_decision.md).

## What changed at the time

The cpp backend used to answer an unhonourable width with a warning and the serialized session:

```
UserWarning: the batch path was asked for (from --max-batch-size 8) but this native module does
not expose QwenBatchScheduler; falling back to serial execution, which serves one request at a time
```

It raised `UnsupportedFeatureError` instead, and the process did not start.

| | Before | At the time |
|---|---|---|
| `--max-batch-size 8` on a build with no `QwenBatchScheduler` | Warning, then one request at a time | `UnsupportedFeatureError` naming the flag and the build |
| `--enable-batching` on the same build | The same warning | The same refusal |
| `--max-batch-size 8` on a checkpoint whose engine the scheduler will not take | Warning, then one request at a time | `UnsupportedFeatureError`, raised **before** the checkpoint is read |
| No width named, either case | Silent fallback; `capabilities.details["scheduler"]` said which session you got | Unchanged |

A width reached the engine through the scheduler's own slot allocation, so "the scheduler could not
be built" and "the width cannot be honoured" were one fact rather than two. Accepting the flag and
serving one request at a time was the third state: the number was parsed, reached `EngineArgs`, and
changed nothing.

The default path was untouched. `pocketllm serve --backend cpp` with no width on a scheduler-less
build ran the serialized session and reported it in `capabilities.details["scheduler"]` — refusing
there would have made the backend unusable on such a build, and nobody asked for anything.

## What to do

Nothing. The backend these flags selected is gone, and `--backend cpp` says so:

```
$ pocketllm serve --model ... --backend cpp
must be one of auto, torch, v41, mimo, xing4, qwen4_exp, got 'cpp'; `cpp` is not in this
distribution: the native engine was removed with the `cpp_engine/` tree. The runtimes are ...
```

Pick a live runtime. None of them owns a batch path — no runtime here exports the scheduler gauges
either — so there is no width to name and no `--max-batch-size` to pass.

## What did not change

The `torch`, `v41`, `mimo`, `xing4` and `qwen4_exp` runtimes refuse a width at parse time, before
anything loads — they declare `supports_batch=False`, so `factory.py` answers them and no adapter's
constructor is reached.