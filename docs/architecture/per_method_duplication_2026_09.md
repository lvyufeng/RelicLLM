# Per-method duplication across the five adapters, 2026-09-29

This is the measurement #442's acceptance rests on, kept because the issue's own numbers were
estimates and the slices that followed kept needing the same question answered again: **which
same-named methods across `pocketllm/backends/*.py` are still the same body?**

The method was `scripts/method_duplication.py`: `ast` per
method, docstrings stripped, comment-only lines dropped, then `difflib.SequenceMatcher` over the
stripped statement lists for every pair of same-named methods in the six modules. What it reported is
the **best** pair per name, the line count of each side, and how many definitions the name has. A
high ratio between one pair out of five says those two agree, not that the family does.

**The script is gone.** It was written against the pre-merge tree — `pocketllm/backends/`, with
`cpp_backend` among the six modules — and by the time it would have been re-run those paths no
longer existed, so it raised `FileNotFoundError` rather than reporting anything. It was deleted
rather than repaired: the table below is the measurement, and the fold it drove (#442) is finished,
so a re-run would answer a question nothing is waiting on. Re-deriving it is a small script
(`ast` + `difflib`) against whatever the adapters are at the time, and a fresh port would report the
current tree rather than this one.

The table below is a snapshot of master `c896925`, on the tree as it was then. The module names in
the `pair` column are the pre-merge ones.

## Where this stood at the end of #442's slices 1–6 (master `c896925`)

| ratio | lines | method | pair |
| ---: | ---: | --- | --- |
| 1.00 | 5/5 | `generate` | v41 ~ mimo |
| 1.00 | 4/4 | `prepare` | mimo ~ xing4 |
| 0.97 | 17/16 | `_run_loop` | mimo ~ xing4 |
| 0.91 | 23/23 | `_eos_tokens` | mimo ~ xing4 |
| 0.90 | 15/16 | `capabilities` | mimo ~ xing4 |
| 0.87 | 15/15 | `_open_tokenizer` | mimo ~ xing4 |
| 0.82 | 34/34 | `_loop` | mimo ~ xing4 |
| 0.81 | 14/13 | `close` | v41 ~ mimo |
| 0.71 | 20/22 | `_start_runtime` | v41 ~ torch |
| 0.67 | 24/27 | `run_worker` | v41 ~ mimo |

Everything the six slices removed is absent from that table: `stream`, `_encode_chat`, `_result`,
`_tokenize`, `_budget`, `_decode`, `_runtime_spec` and the worker program are each one body now.

### Slice 8 (master `839ca6c`): `run_worker`

The one row from that table a later slice opened was `run_worker` (0.67, 24/27, `v41` ~ `mimo`) —
the highest ratio left, and the only item in #442's problem list besides `CppBackend._stream_native`
that a slice could still close. It is now `RankedWorker.run_worker` in
`pocketllm/backends/runtime_engine.py`, with `v41` supplying `_recv_worker_message` (its doorbell,
so the wait happens on the host) and `_WORKER_ABORTS`, and `mimo` supplying `_run_worker_request`
(its payload runner takes one argument, not three) and `_worker_drained` (the barrier that keeps
rank 0 from reaping the group under a worker). The three names `run_worker` resolves to are now
`CppBackend`, `TorchBackend` and the mixin, and the best pair among them is 0.38 — two adapters
with a native worker entry each, which is a different method that shares a name.

The row was worth about a third of what the ratio suggested, and the reason is the general one this
page exists to state: **a ratio counts the statements that matched, not the ones that had to be
invented.** The loop was 28 lines and identical; the four things around it that differed became four
hooks, so the fold is ~20 net lines rather than ~28. What it buys is that the invariants — the two
rank guards, one message shape, `_ensure_loaded` running after the guards, and which exceptions mean
"the group unwound together" — are stated once instead of twice.

## What a ratio does and does not mean

A high ratio is where to look, not a verdict; the two rows that were opened and turned out to be
**one parameter apart** show why:

- `_run_loop` (0.97): the diff is exactly one line, `marks=marks`. `mimo._loop` takes `marks` and
  never reads it — the parameter is dead in the body and alive only in the signature. Nothing in the
  other 16 statements differs.
- `_loop` (0.82): the signature differs (`on_step` vs `marks`), the import inside differs (each
  imports its own runtime's `generate`), and the step predicate differs — `self._step_sync(...)` for
  mimo, one collective per step so every rank agrees, against a lambda folding `on_step` and `stop`
  into the local cancel flag for xing4. That last one is the whole point of the method.

`capabilities` (0.90), `_eos_tokens` (0.91) and `_open_tokenizer` (0.87) are rows where the
difference is small enough to fold behind a per-adapter hook, and all three were **left alone on
purpose**: folding a 15-line body into ~20 shared lines plus a hook is a wash in size, and it hides a
fact (which object carries the eos ids, which path resolves the tokenizer) behind a name instead of
removing it. `close` (0.81) is a fourth: mimo's sends its workers a shutdown broadcast as its first
act and xing4's drops a prefix cache as its last, and those are the two things the method is for.

Two of those rows -- `_eos_tokens` and `_open_tokenizer` -- were later reopened on stronger evidence;
the section at the end of this page records what the fold is and why the reasoning above did not
survive re-measurement.

## What changed for `_eos_tokens` and `_open_tokenizer`, 2026-10-06

Two of those three rows were **reopened and folded** on 2026-10-06, and what changed is the
*evidence* behind the recorded reason, not the measurement. The sentence above
says the fold is "a wash in size" and that it "hides a fact behind a name". Both halves are now known
false, and both were settled by the same thing this page is about: counting what the copies actually
differ by.

**The size claim was wrong, and re-measuring says by how much.** Counting statements the way this
page's table did (docstrings stripped, `ast` over each body): the three `_eos_tokens` copies were 15
statements each and the three `_open_tokenizer` copies 7 each -- **66 in total** -- and 13 of the 15
were the same set-building loop with two `isinstance` branches and the refusal. The only statements
that ever differed were the two-line read of the checkpoint's own config object
(`self._checkpoint.layer`, `self._model.params`, `self._config.text_config`) and the one path
expression. Folded, the same work is **30 statements**: one shared `_open_tokenizer` (7) and
`_eos_tokens` (14) on `RuntimeAdapter`, the two one-expression defaults beside them, and four
one-expression answers across the three adapters -- 23 shared, 7 local, against 66 all-local. The
shared half is the half that carries the refusal and the union, so what each adapter still says is
exactly the fact that was ever local to it.

**The "hides a fact behind a name" claim is the one the two upstreams settle.** The fact in question
is *where a runtime reads its EOS set and its tokenizer from*, and the two projects that serve these
same checkpoint families answer it the other way:

- **vLLM**: `BaseRenderer.get_eos_token_id` (`vllm/renderers/base.py`) is one definition on the base
  class with **zero overrides** in the tree, reading only `self.tokenizer.eos_token_id` and never the
  model config. Its tokenizer resolution is likewise one shared `get_tokenizer()`
  (`vllm/tokenizers/registry.py`), with per-model differences as a small data registry rather than a
  method a model type overrides.
- **SGLang**: `ModelConfig._get_hf_eos_token_id() -> Optional[Set[int]]`
  (`python/sglang/srt/configs/model_config.py`) is one shared method, and it is a *union* -- the HF
  config's `eos_token_id` (scalar or list) unioned with the generation config's -- which is the shape
  this fold settles on, exactly, one `isinstance(ids, int)` branch and all.

Neither project hides "which config object" behind a name because neither has a per-model answer to
hide: the union is the contract, the tokenizer is part of the checkpoint the config describes, and a
runtime that read only one of the two would be the bug. The hook names the one line that genuinely
varies; it does not hide it, because the line is what the hook *is*.

What stayed put, and why: `capabilities` still has no shared body -- its difference is the *content*
of an advertised table, not a computation, and each adapter's row is the declaration itself.
`_eos_token_id` on `v41` is still its own method with its own name, because it is not this union: it
reads one id off the tokenizer to hand the scheduler's `eos_token_id=` kwarg, with no config read and
no refusal. And `v41` still opens its tokenizer inline, over a path its constructor already resolved,
because its load needs the tokenizer *before* the model is built (for the hasher) -- a lifecycle
position the base's `_ensure_loaded`-time hook does not have.

The anti-drift test
(`tests/test_backend_contract.py::test_the_tokenizer_and_eos_resolution_have_one_definition`) pins all
four names to the files allowed to define them, and it treats `v41`'s `_eos_tokens` as the named
exception rather than a silent gap.
