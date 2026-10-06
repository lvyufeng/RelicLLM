"""Prompt rendering tests for the Torch adapter.

These inject a fake serving engine so no checkpoint, CUDA device, or real
runtime is required; only the payload the adapter builds is inspected.
"""

from __future__ import annotations

from relicllm.api import EngineArgs, GenerationRequest, SamplingParams
from relicllm.backends.torch_backend import TorchBackend
from relicllm.encoding.deepseek_v4 import encode_messages


class RecordingServingEngine:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def submit(self, payload: dict) -> dict:
        self.payloads.append(payload)
        return {"content": "ok", "prompt_tokens": 2, "completion_tokens": 1, "finish_reason": "stop"}

    def submit_stream(self, payload: dict):
        self.payloads.append(payload)
        yield {"type": "token", "token_ids": [11]}
        yield {"type": "done", "prompt_tokens": 2, "completion_tokens": [[11]], "finish_reason": "stop"}

    def close(self) -> None:
        pass


class RecordingTokenizer:
    eos_token_id = 1

    def __init__(self) -> None:
        self.encoded: list[str] = []

    def encode(self, text: str) -> list[int]:
        self.encoded.append(text)
        return [11, 12]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(f"<{token}>" for token in token_ids)


class TemplateRecordingTokenizer(RecordingTokenizer):
    chat_template = "{{ messages }}"

    def __init__(self) -> None:
        super().__init__()
        self.template_calls: list[tuple[list[dict], dict]] = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append(([dict(message) for message in messages], dict(kwargs)))
        return [91, 92]


def _backend(tokenizer: RecordingTokenizer) -> tuple[TorchBackend, RecordingServingEngine]:
    engine = RecordingServingEngine()
    backend = TorchBackend(
        EngineArgs(model="model", backend="torch"),
        runtime={"tokenizer": tokenizer, "model_id": "fake-model"},
        serving_engine=engine,
    )
    return backend, engine


def test_chat_metadata_is_rendered_with_the_deepseek_template():
    tokenizer = RecordingTokenizer()
    backend, engine = _backend(tokenizer)
    messages = [{"role": "user", "content": "hi"}]
    request = GenerationRequest(
        prompt="user: hi",
        request_id="req-chat",
        sampling_params=SamplingParams(max_tokens=1),
        metadata={"messages": messages, "thinking_mode": "chat"},
    )

    backend.generate([request])

    assert tokenizer.encoded == [encode_messages(messages, thinking_mode="chat")]
    assert engine.payloads[-1]["messages"] == messages
    assert engine.payloads[-1]["thinking_mode"] == "chat"
    backend.close()


def test_thinking_mode_and_effort_reach_the_template():
    tokenizer = RecordingTokenizer()
    backend, engine = _backend(tokenizer)
    messages = [{"role": "user", "content": "hi"}]
    request = GenerationRequest(
        prompt="user: hi",
        request_id="req-thinking",
        sampling_params=SamplingParams(max_tokens=1),
        metadata={"messages": messages, "thinking_mode": "thinking", "reasoning_effort": "max"},
    )

    backend.generate([request])

    assert tokenizer.encoded == [encode_messages(messages, thinking_mode="thinking", reasoning_effort="max")]
    assert engine.payloads[-1]["reasoning_effort"] == "max"
    backend.close()


def test_tokenizer_owned_template_takes_precedence():
    tokenizer = TemplateRecordingTokenizer()
    backend, _ = _backend(tokenizer)
    messages = [{"role": "user", "content": "hi"}]
    request = GenerationRequest(
        prompt="user: hi",
        request_id="req-owned-template",
        sampling_params=SamplingParams(max_tokens=1),
        metadata={"messages": messages, "thinking_mode": "chat"},
    )

    assert backend._tokenize(request) == [91, 92]
    assert tokenizer.encoded == []
    assert tokenizer.template_calls[0][0] == messages
    assert tokenizer.template_calls[0][1]["add_generation_prompt"] is True
    assert tokenizer.template_calls[0][1]["enable_thinking"] is False
    backend.close()


def test_raw_prompts_are_encoded_unchanged():
    tokenizer = RecordingTokenizer()
    backend, engine = _backend(tokenizer)
    request = GenerationRequest(
        prompt="raw completion prompt",
        request_id="req-raw",
        sampling_params=SamplingParams(max_tokens=1),
    )

    backend.generate([request])

    assert tokenizer.encoded == ["raw completion prompt"]
    assert engine.payloads[-1]["messages"] == [{"role": "user", "content": "raw completion prompt"}]
    backend.close()


def test_prompt_tokens_bypass_the_tokenizer():
    tokenizer = TemplateRecordingTokenizer()
    backend, engine = _backend(tokenizer)
    request = GenerationRequest(
        prompt_tokens=[7, 8],
        request_id="req-tokens",
        sampling_params=SamplingParams(max_tokens=1),
        metadata={"messages": [{"role": "user", "content": "ignored"}]},
    )

    backend.generate([request])

    assert tokenizer.encoded == []
    assert tokenizer.template_calls == []
    assert engine.payloads[-1]["_prompt_ids"] == [7, 8]
    backend.close()


def test_streaming_uses_the_same_prompt_rendering():
    tokenizer = RecordingTokenizer()
    backend, engine = _backend(tokenizer)
    messages = [{"role": "user", "content": "hi"}]
    request = GenerationRequest(
        prompt="user: hi",
        request_id="req-stream",
        sampling_params=SamplingParams(max_tokens=1),
        metadata={"messages": messages, "thinking_mode": "chat"},
    )

    list(backend.stream(request))

    assert tokenizer.encoded[0] == encode_messages(messages, thinking_mode="chat")
    assert engine.payloads[-1]["_prompt_ids"] == [11, 12]
    backend.close()


def test_a_request_without_a_budget_names_the_runtime_default():
    """The legacy runtime's own 512 is written out rather than left off the payload.

    The serving queue's admission check counts that same field against its token budget
    (``relicllm/server/engine.py``), so an absent one would be read there as zero and the request
    would be admitted on a promise the runtime does not keep.
    """
    tokenizer = RecordingTokenizer()
    backend, engine = _backend(tokenizer)
    request = GenerationRequest(prompt="raw prompt", request_id="req-open-ended")

    backend.generate([request])

    assert engine.payloads[-1]["max_tokens"] == 512
    backend.close()


def test_a_budget_is_resolved_against_the_configured_context_first():
    """With a context configured, an absent budget is what the prompt leaves, as in the others."""
    tokenizer = RecordingTokenizer()
    backend, engine = _backend(tokenizer)
    backend.args.max_model_len = 1024

    backend.generate([GenerationRequest(prompt_tokens=[7, 8], request_id="req-open-ended")])
    backend.generate(
        [
            GenerationRequest(
                prompt_tokens=[7, 8],
                request_id="req-named",
                sampling_params=SamplingParams(max_tokens=9),
            )
        ]
    )

    assert [payload["max_tokens"] for payload in engine.payloads] == [1022, 9]
    backend.close()


# ------------------------------------------------------------------- streaming stop, logprobs shape


class WordStreamingEngine(RecordingServingEngine):
    """A stream that hands out one word at a time, the way the real one hands out tokens."""

    def __init__(self, words: list[str]) -> None:
        super().__init__()
        self.words = words

    def submit_stream(self, payload: dict):
        self.payloads.append(payload)
        for word in self.words:
            yield {"type": "token", "token_ids": [11], "text": word}
        yield {"type": "done", "prompt_tokens": 2, "completion_tokens": [[11]], "finish_reason": "length"}


def _word_backend(words: list[str]) -> TorchBackend:
    return TorchBackend(
        EngineArgs(model="model", backend="torch"),
        runtime={"tokenizer": RecordingTokenizer(), "model_id": "fake-model"},
        serving_engine=WordStreamingEngine(words),
    )


def test_a_streamed_stop_string_cuts_the_answer_and_nothing_past_it_goes_out():
    """The mirror of the serial route, which cuts at the marker.

    Without this the streamed answer ran past the marker: `mimo`, `xing4` and `qwen4_exp` matched
    stops in `TokenStreamer`, `v41` in its step hook, and this adapter -- the DeepSeek-V4 runtime --
    nowhere on the streamed path.
    """
    backend = _word_backend(["alpha", "BE", "TA", "gamma"])
    request = GenerationRequest(
        prompt="user: hi",
        request_id="req-stop",
        sampling_params=SamplingParams(stop=("BETA",)),
    )

    events = list(backend.stream(request))

    text = "".join(event.text for event in events)
    assert text == "alpha"
    assert "BETA" not in text
    assert events[-1].finish_reason == "stop"
    backend.close()


def test_a_streamed_answer_with_no_stop_string_keeps_its_own_finish_reason():
    """The field was not asked for, so the loop's own reason is what comes back."""
    backend = _word_backend(["alpha", "beta"])
    request = GenerationRequest(prompt="user: hi", request_id="req-plain")

    events = list(backend.stream(request))

    assert "".join(event.text for event in events) == "alphabeta"
    assert events[-1].finish_reason == "length"
    backend.close()


def test_a_partial_marker_at_the_end_is_text_the_model_wrote():
    """A stop string that never completed is content, not a marker -- the same rule as the other route."""
    backend = _word_backend(["alpha", "BE"])
    request = GenerationRequest(
        prompt="user: hi",
        request_id="req-partial",
        sampling_params=SamplingParams(stop=("BETA",)),
    )

    events = list(backend.stream(request))

    assert "".join(event.text for event in events) == "alphaBE"
    assert events[-1].finish_reason == "length"
    backend.close()


# ------------------------------------------------------------------------------ logprobs response


def test_a_result_mapping_reports_the_ranking_in_the_openai_shape():
    """The adapter's `logprobs` slot is the object a client reads, not the raw float list.

    `_result_from_mapping` used to fall back to `token_logprobs` -- a bare list of floats -- which is
    not the `{"content": [{"token", "logprob", "bytes", "top_logprobs"}]}` shape OpenAI defines. The
    model-side builder now puts that object under `logprobs`, and the receiver's exclusion set keeps
    the three ranking keys out of `metadata`.
    """
    request = GenerationRequest(prompt="user: hi", request_id="req-logprobs")
    ranking = {
        "content": [
            {"token": "a", "logprob": -0.5, "bytes": [97], "top_logprobs": []},
            {"token": "b", "logprob": -1.0, "bytes": [98], "top_logprobs": []},
        ]
    }

    result = TorchBackend._result_from_mapping(
        request,
        {"text": "ab", "token_ids": [1, 2], "logprobs": ranking, "token_logprobs": [-0.5, -1.0]},
    )

    assert result.logprobs == ranking
    assert "token_logprobs" not in result.metadata
    assert "top_logprobs" not in result.metadata
    assert "logprobs" not in result.metadata
