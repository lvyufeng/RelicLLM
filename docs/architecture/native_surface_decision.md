# The native surface decision

What happened to the two APIs that existed only on the native side — the engine's tokenizer and its
constrained-decode builders — when the `cpp` backend was deleted. The decision itself is short: the
tokenizer was **not moved**, and constrained decoding was **dropped while the prompt instruction was
kept**. What takes the length is the evidence, because the second half of "dropped" is a live defect:
`response_format` was accepted, attached to the wrong message, and rendered by no encoder, so json
mode reached the model on no runtime at all.

This page is the record the [issue](https://github.com/lvyufeng/RelicLLM/issues/4) asked for — item 5,
"decide the native tokenizer/structured-output surface" — and it is late by one development cycle:
the decision was carried out in `edbea99` (2026-10-01) and lived only in that commit's message.

## Context

`edbea99` deleted `relicllm/backends/cpp_backend.py` (2,070 lines), the native `BatchScheduler` join
in `runtime_engine.py`, `load_native_module`, `gguf_is_servable` and the `cpp` runtime entry. It also
deleted `tests/test_token_constraint.py` (190 lines), whose first statement was
`pytest.importorskip("pocketllm_cpp")` — a test that had never run on a host without the archived
extension. The four runtimes left — `v41`, `mimo`, `xing4`, `torch` — are pure PyTorch.

Two things in the deleted file were not the backend. They were a *surface*: the native engine's own
tokenizer, and the two factories that turned a JSON schema into a per-token mask.

| Retired piece | What it was |
|---|---|
| `pocketllm_cpp.Tokenizer(path)` | A C++ BPE reader over the checkpoint's vocabulary, loaded in the engine's process. One call site in the Python tree, `cpp_backend.py:1302`; the class is `cpp_engine/include/tokenizer.hpp` (371 lines of implementation) |
| `make_json_object_constraint(tokenizer)` / `make_json_schema_constraint(tokenizer, schema)` | Factories for a `TokenConstraint`: `fill_mask(allowed, begin, end)` sets which of a vocabulary shard's tokens may be sampled next, `accept_token(id)` advances the grammar (`relic-engine/cpp_engine/include/token_constraint.hpp`) |
| `JsonConstraint` (`cpp_engine/core/json_constraint.cpp`, 818 lines) | The JSON grammar state machine behind them: object/array contexts, strings and escapes, numbers, literals |
| `load_native_module`, `gguf_is_servable`, `native_available` | The archive's import surface, which went with the backend |

The C++ *engine* — bindings, scheduler, kernels — lives on in the archived
[relic-engine](https://github.com/lvyufeng/relic-engine). Both halves of this surface went with it.

## Decision 1 — the tokenizer was not moved

**The native `Tokenizer` was the engine's own vocabulary reader, and nothing here needs one.**

The single call — `self._native.Tokenizer(path)` at `cpp_backend.py:1302`, inside
`_constraint_tokenizer()` — carried its own docstring explaining why it existed: a token constraint
is a mask over the vocabulary *piece by piece*, and a piece is what the tokenizer emits rather than
what the vocabulary file stores. A byte-level BPE vocabulary spells a space `Ġ`, so a mask built
anywhere but over the engine's own pieces would refuse every token that continues a word.

So the tokenizer's reader existed to serve the mask, and the mask existed to serve the engine's
per-row sampler. With the engine gone, both readers of that sentence are gone:

- Every runtime here tokenizes its prompts through the checkpoint's own Hugging Face tokenizer —
  `AutoTokenizer.from_pretrained` in `v41_backend.py:609`, `mimo_backend.py:485`,
  `xing4_backend.py:588`, `qwen4_exp_backend.py:319`, and `AutoTokenizer.from_pretrained`
  in the DeepSeek-V4 runtime (`models/deepseek_v4/serving.py:183`, whose V4.1 sibling loads the
  checkpoint's own `encoding/encoding.py`). Decoding back to text uses the same object.
- Nothing else in the tree reads vocabulary pieces. `protocol/logprobs.py` renders a *ranking* the
  runtime hands it; it never touches a vocabulary.

The alternative — porting `core/tokenizer.cpp` (371 lines) into `relicllm/` — would have created a
**second renderer of the prompt**, a different implementation of "what piece does this token mean"
from the HF tokenizer every runtime already uses. Two tokenizers over one checkpoint can disagree
about a merge rule, and the disagreement would show up as a margin the constraint computes over the
wrong pieces. The first renderer is the one the model was trained against; the second one had no
reader left. Not moved.

## Decision 2 — constrained decoding was dropped, the prompt instruction was kept

**What the field buys on this tree is the checkpoint's own instruction to the model, not a mask.**

A `TokenConstraint` could only be built in the engine's process. `fill_mask` was called on the
engine's sampler with a *shard* of a global vocabulary (`qwen_engine.cpp:3572`), and the mask was
applied immediately before the draw. No runtime here has that seam: sampling is a host-side function —
`models/deepseek_v4/generation.py::sample` is six lines of softmax-and-gumbel over a full logits
tensor, and `v41`, `mimo` and `xing4` each have their own equivalent
(`deepseek_v4_1/modules.py::sample`, `mimo_v2/generate.py::sample_token`,
`xing4_0/generate.py::sample_token`). There is no per-token sampler to hang a mask on, and a mask
seam is a kernel question rather than a host one.

What survives is already in every checkpoint's encoder, verbatim — the same template string in
`relicllm/encoding/deepseek_v4.py:49` as in the released checkpoints
(`DeepSeek-V4-Flash-0731/encoding/encoding_dsv4.py:49` and
`DeepSeek-V4.1-Flash/encoding/encoding.py:64`):

```
## Response Format:

You MUST strictly adhere to the following schema to reply:
{schema}
```

That is the prompt-layer path, and it is what `ServedFields.structured_outputs` means on this
tree: **the answer applies the field** — the schema reaches the model as an instruction — **not
that the answer is held to the schema**. Constrained decoding is recorded here as dropped, not
deferred: reintroducing it needs a per-token sampler, which is a relic-core workspace.

| | Retired native path | What ships |
|---|---|---|
| Mechanism | `fill_mask` / `accept_token` over byte-level BPE pieces | `response_format_template` rendered into the prompt |
| Guarantee | Holding to the schema: tokens violating it were masked to `-inf` (or the row failed with `"constraint violation"`) | The model is instructed; the answer may still violate the schema |
| Where | Engine process, per-row sampler | Host, checkpoint's own encoder |

### The defect this decision ran into

The prompt-layer path was **not working** on a conversation anyone would send. `prepare_messages`
(`protocol/chat.py:116-117`) attached `response_format` to the *last* message whose role is in
`{user, developer, system}`, while every encoder renders it only on the `system` branch (and, for
V4's vendored encoder, `developer` as well). The two DeepSeek runtimes are the two that could carry
the field, and on both the attachment and the rendering missed each other. Measured on this host,
against the released checkpoints:

| `messages` | Carrier the field rides on | Rendered? |
|---|---|---|
| `[{user}]` | the user message (index 0) | v4: **no** · v41: **no** |
| `[{system}, {user}]` | the user message (index 1) | v4: **no** · v41: **no** |
| `[{system}]` alone | the system message (index 0) | v4: yes · v41: yes |
| `[{developer}, {user}]` | the user message (index 1) | v4: **no** — even though the developer branch renders the field, it is not the message that carries it |
| a caller who attaches it to a `system` message themselves | the system message | v4: yes · v41: yes |

So on the only conversation shape a chat client ever sends — a system message at most, then a user
turn — the field never reached an encoder. It rendered only when the last instruction-bearing
message *was* the system message, which is a conversation with no user turn at all. It was not a
refusal either: it was accepted and silently dropped, the third state `protocol/contract.py` exists
to prevent — and the one measured here was worse, because the API-level defect hides behind a
correct encoder.

**The fix, and the ruling it carries.** `prepare_messages` attaches the field to a `system` message,
inserting an empty one at index 0 when the conversation has none — the same carrier, and the same
insertion the existing `tools` attachment already uses (`chat.py:105-107`), so a body carrying both
puts both on the one system message. Consequences, recorded:

- **The runtimes that render it are the runtimes that declare it.** `torch` and `v41` declare
  `structured_outputs=True`, because on both the rendered instruction is what the field buys.
  `mimo`, `xing4` and `qwen4_exp` refuse `response_format` by name: their encoders have no such
  branch, and no prompt-layer path exists to declare.
- `{"type": "json_schema"}` rides the same path as `{"type": "json_object"}` and carries the same
  limitation — the schema is an instruction, not a grammar. That is stated where a caller looks, in
  the [API guide](../guides/pocketllm_api.md#structured-outputs).
- The insertion is a **prompt-layer behaviour change**: a conversation that never had a system
  message gains an empty one when (and only when) a request asks for a response format.
- The V4.1 checkpoint encoder raises `NotImplementedError("Unknown role: developer")`, so a
  `developer` message is a 500 there whatever the field does. `prepare_messages` accepts the role,
  which is right for V4 and wrong for V4.1; it is out of this decision's scope and is noted here as
  a measured fact rather than fixed by it.

This is a *systems* decision in the same sense as `edbea99` was one: an honest instruction beats a
silent drop, and the field's contract is re-recorded against what the runtimes actually do rather
than left pointing at a deleted mask builder.

## What the audit says now

`ServedFields` (`relicllm/protocol/contract.py:106-144`) is the declaration `BackendBase.audit_request`
consumes. Before this work the method had **no live override** — the only `ServedFields` constructor
call in the tree died with `CppBackend` — so `audit_request` returned `None` everywhere and a field
the runtime would not apply was accepted and dropped. The next step wires the table from
`backends/capabilities.py` (the place that already answers "can this runtime do X") into that audit,
so "what `/v1/models` says" and "what a request is refused on" are one read. Two entries that
survived the native front end unchanged:

| Field | Kind | Who serves it |
|---|---|---|
| `n` | host fan-out (`relicllm/choices.py`): a request for `n` choices is `n` requests to the runtime, and the choice count is a dispatch rule rather than an adapter capability | every runtime |
| `response_format` | prompt instruction | `torch` and `v41` render it; refused by name on the other three |

The rest of the table — `stop`, `logprobs` and the sampler fields — is where the runtimes differ
from each other, and each row is decided by the runtime rather than by this page. Two are worth
naming because a reader will look for them here: `stop` is served by every runtime (one client
sequence was matched on the streamed route but not the serial one, and now is on both), and
`logprobs` by `torch` alone.

## What is not in this decision

| Not here | Why not |
|---|---|
| A token-level constraint | A per-token sampler is a relic-core/kernel workspace; see Decision 2 |
| The native `Tokenizer` as a Python class | Its only reader was the mask builder; every runtime has a tokenizer. See Decision 1 |
| `protocol/logprobs.py` | Kept, not deleted: it is the ported record of the native `render_logprobs`, the shape OpenAI's endpoint states — while the live V4 rendering is `serving.py::_logprobs_payload`. It has no importers today and is left that way rather than wired to a path that never produces a ranking. See [Log probabilities](../guides/pocketllm_api.md#log-probabilities) |
| `pocketllm_cpp`, `POCKETLLM_*` | The module is archived with the engine; the env-var prefix is a launch-script contract and is not renamed. See `CLAUDE.md` |
