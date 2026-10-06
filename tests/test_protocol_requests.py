from __future__ import annotations

import copy

import pytest

from relicllm.api import SamplingParams
from relicllm.protocol import build_chat_request, render_fallback_prompt


def test_build_chat_request_accepts_openai_sampling_fields_without_explicit_params():
    request = build_chat_request(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 5,
            "temperature": 0.4,
            "stop": "END",
        }
    )

    assert request.sampling_params.max_tokens == 5
    assert request.sampling_params.temperature == 0.4
    assert request.sampling_params.stop == ("END",)

def test_build_chat_request_preserves_normalized_chat_fields_and_sampling():
    body = {
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": [{"type": "text", "text": "What is 2+2?"}]},
        ],
        "reasoning": {"effort": "high"},
        "tools": [{"type": "function", "function": {"name": "calculator"}}],
        "tool_choice": "required",
        "response_format": {"type": "json_object"},
    }

    request = build_chat_request(body, SamplingParams(max_tokens=7), request_id="chat-7")

    assert request.request_id == "chat-7"
    assert request.sampling_params.max_tokens == 7
    assert request.metadata["thinking_mode"] == "thinking"
    assert request.metadata["reasoning_effort"] == "high"
    assert request.metadata["response_format"] == {"type": "json_object"}
    assert request.metadata["tools"] == body["tools"]
    assert request.metadata["messages"][0]["tools"] == body["tools"]
    assert "must call at least one available tool" in request.metadata["messages"][-1]["content"]
    assert request.prompt == render_fallback_prompt(request.metadata["messages"])


def test_build_chat_request_does_not_mutate_nested_caller_data():
    body = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Weather?"}],
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "parameters": {"properties": {"city": {"type": "string"}}},
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "weather"}},
    }
    original = copy.deepcopy(body)

    request = build_chat_request(body, SamplingParams(max_tokens=2))

    assert body == original
    request.metadata["messages"][0]["content"] = "changed"
    request.metadata["tools"][0]["function"]["name"] = "changed"
    assert body == original


def test_build_chat_request_uses_body_id_and_normalizes_sampling_when_omitted():
    request = build_chat_request({
        "messages": [{"role": "user", "content": "hi"}],
        "request_id": "body-id",
        "max_completion_tokens": 3,
        "temperature": 0.25,
    })

    assert request.request_id == "body-id"
    assert request.sampling_params.max_tokens == 3
    assert request.sampling_params.temperature == 0.25


def test_build_chat_request_generates_id_when_not_supplied():
    request = build_chat_request(
        {"messages": [{"role": "user", "content": "hi"}]},
        SamplingParams(max_tokens=1),
    )

    assert request.request_id.startswith("req-")


def test_build_completion_request_preserves_raw_prompt():
    from relicllm.protocol import build_completion_request

    request = build_completion_request({"prompt": "raw", "max_tokens": 2}, request_id="completion-id")

    assert request.prompt == "raw"
    assert request.request_id == "completion-id"
    assert request.metadata == {"thinking_mode": "chat", "stream_options": {}, "response_format": None}
    assert request.sampling_params.max_tokens == 2


def test_build_completion_request_does_not_mutate_body():
    from relicllm.protocol import build_completion_request

    body = {"prompt": ["a", "b"], "response_format": {"type": "text"}}
    original = copy.deepcopy(body)
    build_completion_request(body)
    assert body == original


def test_sampling_params_are_copied_for_request_ownership():
    params = SamplingParams(max_tokens=2, extra={"nested": {"value": 1}})
    request = build_chat_request({"messages": [{"role": "user", "content": "hi"}]}, params)

    assert request.sampling_params is not params
    request.sampling_params.extra["nested"]["value"] = 2
    assert params.extra["nested"]["value"] == 1


@pytest.mark.parametrize(
    "messages",
    [[], ["not an object"], "not a message list"],
)
def test_build_chat_request_rejects_invalid_messages(messages):
    with pytest.raises(ValueError):
        build_chat_request(
            {"messages": messages},
            SamplingParams(max_tokens=1),
        )


def test_add_generation_prompt_is_carried_only_when_a_request_overrides_it():
    """The flag decides whether the prompt ends with the assistant header the model answers into.

    Carried only when it overrides the default, which is what every backend already does -- the same
    way ``reasoning_effort`` and ``tools`` are -- so a request that does not mention it carries no
    extra key and a backend reads the flag with a default of true. A completion carries none either
    way: its prompt is literal text with no template to add an assistant header to.
    """
    default = build_chat_request({"messages": [{"role": "user", "content": "hi"}]})
    assert "add_generation_prompt" not in default.metadata

    off = build_chat_request(
        {"messages": [{"role": "user", "content": "hi"}], "add_generation_prompt": False}
    )
    assert off.metadata["add_generation_prompt"] is False

    from relicllm.protocol import build_completion_request

    completion = build_completion_request({"prompt": "raw"})
    assert "add_generation_prompt" not in completion.metadata


def test_a_body_key_this_class_holds_no_field_for_becomes_an_extra_option():
    """Only the runtime's own knobs, not the request itself.

    ``from_openai`` is handed the whole client body, so "everything the class does not hold" was
    every key the chat request carries too -- and those became generation options. The failure was
    quiet: ``model``, ``messages``, the tool definitions and ``stream`` were merged into the options
    mapping a runtime looks a sampling knob up in, next to the ones it knows. A name without an
    OpenAI spelling still becomes an option, which is what this field is for.
    """
    params = SamplingParams.from_openai(
        {
            "model": "some-checkpoint",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "max_tokens": 8,
            "tools": [{"type": "function", "function": {"name": "f"}}],
            "tool_choice": "auto",
            "user": "a-user-id",
            "chat_template_kwargs": {"enable_answer_prefix": True},
        }
    )

    assert params.extra == {"chat_template_kwargs": {"enable_answer_prefix": True}}
    for correspondence in ("model", "messages", "stream", "user", "tool_choice"):
        assert correspondence not in params.to_generation_options()


def test_a_refusable_field_is_never_accepted_as_an_extra_option():
    """Routing around the audit is not a use of ``extra``.

    ``logit_bias`` is a field this backend cannot serve and ``audit`` is what reports it, so a name
    in the refusal surface must not be classified as a runtime option: accepting it here would hand
    the sampler a value the field contract had just declined to answer for. The refusal itself is
    pinned where the audit lives; this pins that the door is not open from the other side.
    """
    body = {"logit_bias": {"1234": -100}, "best_of": 2, "echo": True}
    params = SamplingParams.from_openai(body)

    assert params.extra == {}
    assert params.sampling_body() == {}


def test_one_method_answers_what_a_body_would_say_for_both_entry_points():
    """``sampling_body`` is the single bridge to ``audit``, so the two doors cannot disagree.

    The library path used to spell this out in ``engine.py`` while the HTTP path spelled the same
    fields out again in ``from_openai``. Two writers of one audit input is how a field becomes
    refusable on one route and silent on the other; both call this now.
    """
    params = SamplingParams(
        max_tokens=12,
        n=2,
        stop=("END",),
        logprobs=True,
        top_logprobs=3,
        frequency_penalty=0.5,
        response_format={"type": "json_object"},
        extra={"chat_template_kwargs": {"x": 1}},
    )
    body = params.sampling_body(stream=True)

    assert body == {
        "stream": True,
        "n": 2,
        "stop": ["END"],
        "logprobs": True,
        "top_logprobs": 3,
        "frequency_penalty": 0.5,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"x": 1},
    }
    # The defaults a caller never named are absent, so an audit of this request is not an audit of
    # fields it did not use -- and never carries the budget, which no runtime refuses.
    for unasked in ("temperature", "min_p", "presence_penalty", "repetition_penalty", "max_tokens"):
        assert unasked not in body
    # The only thing the stream keyword changes is the name the audit reads for it.
    assert params.sampling_body() == {k: v for k, v in body.items() if k != "stream"}
    assert SamplingParams().sampling_body() == {}
