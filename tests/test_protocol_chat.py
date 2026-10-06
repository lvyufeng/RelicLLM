from __future__ import annotations

import pytest

from relicllm.protocol import (
    ChatRequest,
    apply_stop_to_text,
    normalize_content,
    normalize_tool_calls,
    prepare_messages,
    render_fallback_prompt,
    thinking_config,
)


def test_normalize_content_flattens_text_blocks_and_marks_others():
    assert normalize_content("plain") == "plain"
    assert normalize_content([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "a\nb"
    assert normalize_content([{"type": "image_url"}]) == "[Unsupported image_url]"
    assert normalize_content(None) == ""


def test_prepare_messages_attaches_tools_and_required_instruction():
    body = {
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        "tools": [{"type": "function", "function": {"name": "weather", "parameters": {}}}],
        "tool_choice": "required",
    }

    messages = prepare_messages(body)

    # A system turn is inserted to carry tools, and the instruction lands on the
    # last instruction-bearing message.
    assert messages[0]["role"] == "system"
    assert messages[0]["tools"] == body["tools"]
    assert messages[-1]["content"].startswith("hi")
    assert "must call at least one available tool" in messages[-1]["content"]


def test_prepare_messages_rejects_unknown_named_tool():
    body = {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "weather"}}],
        "tool_choice": {"type": "function", "function": {"name": "missing"}},
    }

    with pytest.raises(ValueError, match="unknown tool"):
        prepare_messages(body)


# ----------------------------------------------------------------------------------- response format


def test_a_response_format_rides_a_system_message_and_inserts_one_when_absent():
    """The carrier is the encoder's, not the conversation's.

    Both DeepSeek encoders render the schema block only under ``role == "system"``, so attaching the
    field to the last instruction-bearing message -- which is the user turn in every conversation a
    chat client sends -- meant the field never reached the prompt. See
    ``docs/architecture/native_surface_decision.md``.
    """
    body = {
        "messages": [{"role": "user", "content": "hi"}],
        "response_format": {"type": "json_object"},
    }

    messages = prepare_messages(body)

    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[0]["response_format"] == {"type": "json_object"}
    assert "response_format" not in messages[1]
    # The inserted carrier is empty: the schema block is appended after the system content, and a
    # placeholder sentence would be an instruction the caller never wrote.
    assert messages[0]["content"] == ""


def test_a_conversation_that_has_a_system_message_keeps_it_and_carries_both_fields():
    tools = [{"type": "function", "function": {"name": "weather", "parameters": {}}}]
    body = {
        "messages": [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "hi"},
        ],
        "tools": tools,
        "response_format": {"type": "json_object"},
    }

    messages = prepare_messages(body)

    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[0]["content"] == "Be terse."
    assert messages[0]["tools"] == tools
    assert messages[0]["response_format"] == {"type": "json_object"}


def test_the_response_format_and_the_tool_list_share_one_inserted_carrier():
    tools = [{"type": "function", "function": {"name": "weather", "parameters": {}}}]
    body = {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": tools,
        "response_format": {"type": "json_schema", "json_schema": {"name": "W", "schema": {}}},
    }

    messages = prepare_messages(body)

    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[0]["tools"] == tools
    assert messages[0]["response_format"]["type"] == "json_schema"


def test_a_text_response_format_is_not_carried_at_all():
    """``{"type": "text"}`` is the value an OpenAI client sends by default and asks for nothing.

    Rendering it would put a schema block in the prompt whose schema is ``{"type": "text"}`` -- an
    instruction the caller did not write -- and inserting a carrier for it would change the prompt
    of a request that asked for no change.
    """
    body = {
        "messages": [{"role": "user", "content": "hi"}],
        "response_format": {"type": "text"},
    }

    messages = prepare_messages(body)

    assert [message["role"] for message in messages] == ["user"]
    assert "response_format" not in messages[0]


def test_a_developer_message_does_not_carry_the_response_format():
    """V4.1's checkpoint encoder raises on the role, so the field must not take it there either.

    The role is accepted (V4's own encoder renders it) and the carrier is a system message: a
    conversation whose last instruction-bearing turn is a developer one still gets an inserted
    system message rather than a schema block the encoder would refuse.
    """
    body = {
        "messages": [
            {"role": "developer", "content": "Be terse."},
            {"role": "user", "content": "hi"},
        ],
        "response_format": {"type": "json_object"},
    }

    messages = prepare_messages(body)

    assert [message["role"] for message in messages] == ["system", "developer", "user"]
    assert messages[0]["response_format"] == {"type": "json_object"}
    assert "response_format" not in messages[1]


def test_the_carrier_renders_the_schema_block_the_two_deepseek_encoders_read():
    """The end the attachment exists for: the encoder sees the block.

    Pinned against the encoder rather than against the message dict, because the defect this
    replaced was precisely a carrier that looked right in the intermediate value and rendered
    nothing.
    """
    from relicllm.encoding.deepseek_v4 import encode_messages

    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": False,
    }
    body = {
        "messages": [{"role": "user", "content": "Capital of France?"}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "City", "schema": schema}},
    }

    rendered = encode_messages(prepare_messages(body), thinking_mode="chat")

    assert "## Response Format:" in rendered
    assert "You MUST strictly adhere to the following schema to reply:" in rendered
    assert '"city"' in rendered
    assert "Capital of France?" in rendered


def test_a_malformed_response_format_is_left_for_the_audit_to_refuse():
    """Normalization does not double as validation.

    ``contract.structured_output_spec`` is what reads the value, and a value it cannot read is a
    400 from the audit before dispatch -- so the normalization layer leaves the conversation alone
    rather than attaching something no encoder should render.
    """
    body = {"messages": [{"role": "user", "content": "hi"}], "response_format": {"type": "bogus"}}

    messages = prepare_messages(body)

    assert [message["role"] for message in messages] == ["user"]
    assert "response_format" not in messages[0]


def test_prepare_messages_requires_a_non_empty_list():
    with pytest.raises(ValueError, match="non-empty list"):
        prepare_messages({"messages": []})
    with pytest.raises(ValueError, match="each message must be an object"):
        prepare_messages({"messages": ["hi"]})


def test_thinking_config_reads_reasoning_effort_from_either_field():
    assert thinking_config({}) == ("chat", None)
    assert thinking_config({"reasoning_effort": "high"}) == ("thinking", "high")
    assert thinking_config({"reasoning": {"effort": "max"}}) == ("thinking", "max")
    # An unrecognized effort still selects thinking mode but drops the value.
    assert thinking_config({"reasoning_effort": "turbo"}) == ("thinking", None)


def test_normalize_tool_calls_serializes_arguments_and_assigns_ids():
    calls = normalize_tool_calls([{"function": {"name": "weather", "arguments": {"city": "sh"}}}])

    assert calls[0]["type"] == "function"
    assert calls[0]["id"].startswith("call_")
    assert calls[0]["function"] == {"name": "weather", "arguments": '{"city": "sh"}'}
    assert normalize_tool_calls([{"function": {}}]) == []


def test_apply_stop_to_text_truncates_at_the_earliest_match():
    assert apply_stop_to_text("abcdef", ["cd", "ef"]) == ("ab", True)
    assert apply_stop_to_text("abcdef", "zz") == ("abcdef", False)
    assert apply_stop_to_text("abcdef", None) == ("abcdef", False)


def test_chat_request_carries_normalized_messages_as_metadata():
    request = ChatRequest.from_body({
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        "reasoning_effort": "low",
    })

    metadata = request.metadata()
    assert metadata["thinking_mode"] == "thinking"
    assert metadata["reasoning_effort"] == "low"
    assert metadata["messages"] == [{"role": "user", "content": "hi"}]


def test_chat_request_validates_completion_prompts():
    assert ChatRequest.from_body({"prompt": ["a", "b"]}, completion=True).prompt == "ab"
    with pytest.raises(ValueError, match="prompt must be a non-empty string"):
        ChatRequest.from_body({"prompt": ""}, completion=True)
    with pytest.raises(ValueError, match="prompt list must contain only strings"):
        ChatRequest.from_body({"prompt": [1]}, completion=True)
    with pytest.raises(ValueError, match="stream_options must be an object"):
        ChatRequest.from_body({"prompt": "hi", "stream_options": 1}, completion=True)


def test_render_fallback_prompt_is_deterministic():
    text = render_fallback_prompt([
        {"role": "system", "content": "s"},
        {"role": "user", "content": [{"type": "text", "text": "u"}]},
    ])

    assert text == "system: s\nuser: u"
