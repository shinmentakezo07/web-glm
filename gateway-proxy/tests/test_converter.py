"""Unit tests for the OpenAI <-> AI SDK v3 converter.

Wire-format assertions ensure converter.py and server.py always agree on
the exact shape sent to / received from the Vercel AI Gateway.
"""

import json

from converter import (
    _openai_content_to_v3_parts,
    _openai_tool_call_to_v3,
    _normalize_tool_choice,
    openai_to_v3,
    v3_to_openai,
    v3_stream_to_openai,
    v3_stream_iter,
    v3_sse_stream_to_openai,
    _v3_finish_reason,
)


# =====================================================================
# Content-part conversion
# =====================================================================


class TestContentParts:
    def test_none_becomes_empty_text(self):
        assert _openai_content_to_v3_parts(None) == [{"type": "text", "text": ""}]

    def test_empty_string(self):
        assert _openai_content_to_v3_parts("") == [{"type": "text", "text": ""}]

    def test_plain_string(self):
        assert _openai_content_to_v3_parts("hello") == [{"type": "text", "text": "hello"}]

    def test_list_of_strings(self):
        assert _openai_content_to_v3_parts(["a", "b"]) == [
            {"type": "text", "text": "a"},
            {"type": "text", "text": "b"},
        ]

    def test_mixed_list_with_image(self):
        parts = _openai_content_to_v3_parts([
            {"type": "text", "text": "see image"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ])
        assert parts == [
            {"type": "text", "text": "see image"},
            {"type": "image", "image": "data:image/png;base64,abc"},
        ]


# =====================================================================
# Tool-call conversion (wire format)
# =====================================================================


class TestToolCallConversion:
    def test_string_args_passed_through(self):
        tc = {"id": "call_1", "type": "function", "function": {"name": "read", "arguments": '{"path":"x"}'}}
        result = _openai_tool_call_to_v3(tc)
        assert result == {
            "toolCallId": "call_1",
            "toolName": "read",
            "input": '{"path":"x"}',
        }

    def test_object_args_stringified(self):
        tc = {"id": "call_2", "type": "function", "function": {"name": "write", "arguments": {"path": "x", "content": "y"}}}
        result = _openai_tool_call_to_v3(tc)
        assert result["input"] == '{"path": "x", "content": "y"}'
        assert isinstance(result["input"], str)

    def test_invalid_args_defaults_to_empty(self):
        tc = {"id": "call_3", "type": "function", "function": {"name": "x", "arguments": None}}
        result = _openai_tool_call_to_v3(tc)
        assert result["input"] == "{}"


# =====================================================================
# openai_to_v3 full conversion
# =====================================================================


class TestOpenAIToV3:
    def test_basic_user_message(self):
        body = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}
        result = openai_to_v3(body)
        assert result["prompt"][0] == {"role": "user", "content": [{"type": "text", "text": "hi"}]}
        assert result["tools"] == []
        assert result["toolChoice"] == {"type": "auto"}

    def test_system_message_flattened(self):
        body = {"messages": [{"role": "system", "content": "be nice"}]}
        result = openai_to_v3(body)
        assert result["prompt"][0] == {"role": "system", "content": "be nice"}

    def test_system_multi_part(self):
        body = {"messages": [{"role": "system", "content": [
            {"type": "text", "text": "a"},
            {"type": "text", "text": "b"},
        ]}]}
        result = openai_to_v3(body)
        assert result["prompt"][0]["content"] == "ab"

    def test_tools_flattened_with_inputSchema(self):
        body = {
            "messages": [{"role": "user", "content": "x"}],
            "tools": [{
                "type": "function",
                "function": {"name": "read_file", "description": "reads", "parameters": {"type": "object"}},
            }],
        }
        result = openai_to_v3(body)
        assert result["tools"] == [{
            "type": "function",
            "name": "read_file",
            "description": "reads",
            "inputSchema": {"type": "object"},
        }]

    def test_assistant_tool_calls_use_top_level_toolCalls(self):
        body = {
            "messages": [
                {"role": "user", "content": "run"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "calc", "arguments": '{"a":1}'},
                    }],
                },
            ],
        }
        result = openai_to_v3(body)
        assistant_msg = result["prompt"][1]
        assert "toolCalls" in assistant_msg
        assert assistant_msg["toolCalls"] == [{
            "toolCallId": "call_1",
            "toolName": "calc",
            "input": '{"a":1}',
        }]
        # Must NOT be inside content as parts.
        assert not any(p.get("type") == "tool-call" for p in assistant_msg["content"])

    def test_tool_result_name_backfilled_from_prior_assistant(self):
        body = {
            "messages": [
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "my_tool", "arguments": "{}"},
                    }],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "result",
                },
            ],
        }
        result = openai_to_v3(body)
        tool_msg = result["prompt"][2]
        part = tool_msg["content"][0]
        assert part["type"] == "tool-result"
        assert part["toolName"] == "my_tool"
        assert part["output"] == {"type": "text", "value": "result"}

    def test_tool_result_no_prior_assistant_empty_name(self):
        body = {"messages": [{"role": "tool", "tool_call_id": "x", "content": "y"}]}
        result = openai_to_v3(body)
        assert result["prompt"][0]["content"][0]["toolName"] == ""

    def test_param_passthrough(self):
        body = {"messages": [{"role": "user", "content": "x"}], "temperature": 0.7, "max_tokens": 50, "top_p": 0.9, "stop": "end"}
        result = openai_to_v3(body)
        assert result["temperature"] == 0.7
        assert result["maxOutputTokens"] == 50
        assert result["topP"] == 0.9
        assert result["stopSequences"] == ["end"]

    def test_stop_string_to_list(self):
        body = {"messages": [{"role": "user", "content": "x"}], "stop": "stop_word"}
        result = openai_to_v3(body)
        assert result["stopSequences"] == ["stop_word"]

    def test_maxOutputTokens_passed_through(self):
        body = {"messages": [{"role": "user", "content": "x"}], "maxOutputTokens": 100}
        result = openai_to_v3(body)
        assert result["maxOutputTokens"] == 100

    def test_tool_choice_string_auto_normalized(self):
        body = {"messages": [{"role": "user", "content": "x"}], "tool_choice": "auto"}
        assert openai_to_v3(body)["toolChoice"] == {"type": "auto"}

    def test_tool_choice_string_none_normalized(self):
        body = {"messages": [{"role": "user", "content": "x"}], "tool_choice": "none"}
        assert openai_to_v3(body)["toolChoice"] == {"type": "none"}

    def test_tool_choice_string_required_normalized(self):
        body = {"messages": [{"role": "user", "content": "x"}], "tool_choice": "required"}
        assert openai_to_v3(body)["toolChoice"] == {"type": "required"}

    def test_tool_choice_unknown_string_defaults_auto(self):
        assert _normalize_tool_choice("bogus") == {"type": "auto"}

    def test_tool_choice_function_shape_to_tool(self):
        tc = {"type": "function", "function": {"name": "my_tool"}}
        assert _normalize_tool_choice(tc) == {"type": "tool", "toolName": "my_tool"}

    def test_tool_choice_v3_object_passthrough(self):
        tc = {"type": "auto"}
        assert _normalize_tool_choice(tc) == {"type": "auto"}

    def test_tool_choice_default_auto(self):
        body = {"messages": [{"role": "user", "content": "x"}]}
        assert openai_to_v3(body)["toolChoice"] == {"type": "auto"}


# =====================================================================
# v3_to_openai (non-streaming response)
# =====================================================================


class TestV3ToOpenAI:
    def test_plain_text(self):
        v3 = {"content": [{"type": "text", "text": "hello world"}], "finishReason": "stop"}
        result = v3_to_openai(v3, model="gpt-4")
        choice = result["choices"][0]
        assert choice["message"]["role"] == "assistant"
        assert choice["message"]["content"] == "hello world"
        assert choice["finish_reason"] == "stop"
        assert result["object"] == "chat.completion"

    def test_tool_calls_from_top_level_toolCalls(self):
        v3 = {
            "content": [],
            "toolCalls": [{
                "toolCallId": "call_1",
                "toolName": "calc",
                "input": '{"a":1}',
            }],
            "finishReason": "tool_calls",
        }
        result = v3_to_openai(v3, model="gpt-4")
        msg = result["choices"][0]["message"]
        assert msg["content"] is None
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["function"]["name"] == "calc"
        assert msg["tool_calls"][0]["id"] == "call_1"

    def test_tool_calls_from_content_parts(self):
        v3 = {
            "content": [{
                "type": "tool-call",
                "toolCallId": "call_2",
                "toolName": "web_fetch",
                "input": {"url": "https://example.com"},
            }],
            "finishReason": "tool-calls",
        }
        result = v3_to_openai(v3)
        tc = result["choices"][0]["message"]["tool_calls"][0]
        assert tc["function"]["name"] == "web_fetch"
        assert tc["function"]["arguments"] == '{"url": "https://example.com"}'

    def test_finish_reason_map(self):
        assert _v3_finish_reason("stop") == "stop"
        assert _v3_finish_reason("length") == "length"
        assert _v3_finish_reason("tool-calls") == "tool_calls"
        assert _v3_finish_reason("tool_calls") == "tool_calls"
        assert _v3_finish_reason("content-filter") == "content_filter"
        assert _v3_finish_reason("content_filter") == "content_filter"
        assert _v3_finish_reason(None) == "stop"
        assert _v3_finish_reason("unknown") == "stop"

    def test_finish_reason_dict(self):
        assert _v3_finish_reason({"unified": "length"}) == "length"

    def test_usage_mapping(self):
        v3 = {
            "content": [{"type": "text", "text": "hi"}],
            "finishReason": "stop",
            "usage": {"inputTokens": {"total": 10}, "outputTokens": {"total": 5}},
        }
        result = v3_to_openai(v3)
        assert result["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    def test_model_passthrough(self):
        v3 = {"content": [], "finishReason": "stop"}
        result = v3_to_openai(v3, model="my-model")
        assert result["model"] == "my-model"


# =====================================================================
# Finish reason mapping (explicit edge cases)
# =====================================================================


class TestFinishReason:
    def test_snake_case(self):
        assert _v3_finish_reason("content_filter") == "content_filter"

    def test_hyphen_case(self):
        assert _v3_finish_reason("content-filter") == "content_filter"


# =====================================================================
# Streaming conversion
# =====================================================================


class TestStreaming:
    def test_basic_stream(self):
        events = [
            {"type": "text-delta", "delta": "hel"},
            {"type": "text-delta", "delta": "lo"},
            {"type": "finish", "finishReason": "stop", "usage": {"inputTokens": {"total": 3}, "outputTokens": {"total": 2}}},
        ]
        sse = v3_stream_to_openai(events, model="gpt-4")

        lines = [line for line in sse.split("\n\n") if line.startswith("data:")]
        role_chunk = json.loads(lines[0][6:])
        assert role_chunk["choices"][0]["delta"] == {"role": "assistant"}
        assert role_chunk["model"] == "gpt-4"
        assert "hel" in sse
        assert "lo" in sse
        assert "data: [DONE]" in sse

    def test_stream_multiple_tool_calls_sequential_indices(self):
        events = [
            {"type": "tool-call", "toolCallId": "call_1", "toolName": "read", "input": "{}"},
            {"type": "tool-call", "toolCallId": "call_2", "toolName": "write", "input": "{}"},
            {"type": "finish", "finishReason": "tool-calls"},
        ]
        sse = v3_stream_to_openai(events)

        tool_chunks = []
        for line in sse.split("\n\n"):
            if line.startswith("data: "):
                try:
                    obj = json.loads(line[6:])
                    delta = obj["choices"][0]["delta"]
                    if "tool_calls" in delta:
                        tool_chunks.append(delta["tool_calls"][0]["index"])
                except (json.JSONDecodeError, KeyError, IndexError):
                    pass

        assert tool_chunks == [0, 1], f"expected sequential indices, got {tool_chunks}"

    def test_stream_tool_input_delta_accumulation(self):
        delta1 = '{"a"},'
        delta2 = '"b":2}'
        events = [
            {"type": "tool-input-delta", "toolCallId": "call_1", "delta": delta1},
            {"type": "tool-input-delta", "toolCallId": "call_1", "delta": delta2},
            {"type": "tool-call", "toolCallId": "call_1", "toolName": "calc"},
            {"type": "finish", "finishReason": "tool-calls"},
        ]
        sse = v3_stream_to_openai(events)
        for line in sse.split("\n\n"):
            if line.startswith("data: "):
                try:
                    obj = json.loads(line[6:])
                    tc = obj["choices"][0]["delta"].get("tool_calls", [])
                    if tc and tc[0].get("function", {}).get("name") == "calc":
                        assert tc[0]["function"]["arguments"] == delta1 + delta2
                        return
                except (json.JSONDecodeError, KeyError):
                    pass
        assert False, "tool-call chunk not found"

    def test_stream_iter_generates_same_chunks(self):
        events = [
            {"type": "text-delta", "delta": "hi"},
            {"type": "finish", "finishReason": "stop"},
        ]
        chunks = list(v3_stream_iter(iter(events)))
        sse = "".join(chunks)
        assert "hi" in sse
        assert "data: [DONE]" in sse


# =====================================================================
# Non-streaming SSE collection
# =====================================================================


class TestNonStreamingCollection:
    def test_collects_text_and_usage(self):
        events = [
            {"type": "text-delta", "delta": "hello"},
            {"type": "text-delta", "delta": " world"},
            {"type": "finish", "finishReason": "stop", "usage": {"inputTokens": {"total": 2}, "outputTokens": {"total": 2}}},
        ]
        result = v3_sse_stream_to_openai(iter(events), model="gpt-4")
        choice = result["choices"][0]
        assert choice["message"]["content"] == "hello world"
        assert choice["finish_reason"] == "stop"
        assert result["usage"] == {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4}

    def test_collects_tool_calls(self):
        events = [
            {"type": "tool-call", "toolCallId": "call_1", "toolName": "run", "input": "{}"},
            {"type": "finish", "finishReason": "tool-calls"},
        ]
        result = v3_sse_stream_to_openai(iter(events))
        msg = result["choices"][0]["message"]
        assert msg["content"] is None
        assert msg["tool_calls"][0]["function"]["name"] == "run"

    def test_tool_input_object_stringified(self):
        events = [
            {"type": "tool-call", "toolCallId": "call_1", "toolName": "f", "input": {"key": "val"}},
            {"type": "finish", "finishReason": "stop"},
        ]
        result = v3_sse_stream_to_openai(iter(events))
        args = result["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        assert isinstance(args, str)
        assert json.loads(args) == {"key": "val"}

    def test_empty_response(self):
        result = v3_sse_stream_to_openai(iter([]))
        assert result["choices"][0]["message"]["content"] == ""
        assert result["choices"][0]["finish_reason"] == "stop"
