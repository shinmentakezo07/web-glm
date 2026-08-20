"""Unit tests for the OpenAI <-> AI SDK v3 converter.

Wire-format assertions ensure converter.py and server.py always agree on
the exact shape sent to / received from the Vercel AI Gateway (fx wire
format: tool calls as content parts with raw-JSON `input`).
"""

import asyncio
import json

from converter import (
    _openai_content_to_v3_parts,
    _openai_tool_call_to_v3,
    _openai_tool_msg_to_v3,
    _normalize_tool_choice,
    _v3_finish_reason,
    openai_to_v3,
    v3_to_openai,
    v3_stream_to_openai,
    v3_stream_iter,
    v3_sse_stream_to_openai,
    validate_tool_history,
    responses_input_to_messages,
    openai_to_responses,
    openai_chunk_to_responses_sse,
    _ResponsesStreamState,
    v3_stream_to_responses_sse,
)


def async_iter(events):
    async def gen():
        for e in events:
            yield e
    return gen()


def run_stream(events, **kwargs):
    """Collect an async stream generator into a string (no async test dep)."""
    return asyncio.run(_collect(events, kwargs))


async def _collect(events, kwargs):
    chunks = []
    async for c in v3_stream_iter(async_iter(events), **kwargs):
        chunks.append(c)
    return "".join(chunks)


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
# Tool-call conversion (fx wire format: content parts, raw JSON input)
# =====================================================================


class TestToolCallConversion:
    def test_string_args_parsed_to_raw_object(self):
        tc = {"id": "call_1", "type": "function", "function": {"name": "read", "arguments": '{"path":"x"}'}}
        result = _openai_tool_call_to_v3(tc)
        assert result == {
            "type": "tool-call",
            "toolCallId": "call_1",
            "toolName": "read",
            "input": {"path": "x"},
        }

    def test_object_args_passed_as_object(self):
        tc = {"id": "call_2", "type": "function", "function": {"name": "write", "arguments": {"path": "x", "content": "y"}}}
        result = _openai_tool_call_to_v3(tc)
        assert result["input"] == {"path": "x", "content": "y"}
        assert isinstance(result["input"], dict)

    def test_invalid_args_defaults_to_empty_object(self):
        tc = {"id": "call_3", "type": "function", "function": {"name": "x", "arguments": None}}
        result = _openai_tool_call_to_v3(tc)
        assert result["input"] == {}

    def test_missing_type_treated_as_function(self):
        tc = {"id": "call_4", "function": {"name": "f", "arguments": "{}"}}
        result = _openai_tool_call_to_v3(tc)
        assert result["type"] == "tool-call"
        assert result["toolName"] == "f"


class TestToolMsgConversion:
    def test_string_output(self):
        msg = {"role": "tool", "tool_call_id": "call_1", "content": "result text"}
        result = _openai_tool_msg_to_v3(msg)
        assert result["role"] == "tool"
        assert result["content"] == [{
            "type": "tool-result",
            "toolCallId": "call_1",
            "toolName": "unknown",
            "output": {"type": "text", "value": "result text"},
        }]

    def test_default_tool_name_unknown(self):
        msg = {"role": "tool", "tool_call_id": "call_1", "content": "x"}
        assert _openai_tool_msg_to_v3(msg)["content"][0]["toolName"] == "unknown"

    def test_name_from_message(self):
        msg = {"role": "tool", "tool_call_id": "call_1", "name": "my_tool", "content": "x"}
        assert _openai_tool_msg_to_v3(msg)["content"][0]["toolName"] == "my_tool"

    def test_text_part_list(self):
        msg = {"role": "tool", "tool_call_id": "call_1",
               "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}
        result = _openai_tool_msg_to_v3(msg)
        assert result["content"][0]["output"]["value"] == "ab"


class TestNormalizeToolChoice:
    def test_string_auto(self):
        assert _normalize_tool_choice("auto") == {"type": "auto"}

    def test_string_none(self):
        assert _normalize_tool_choice("none") == {"type": "none"}

    def test_string_required(self):
        assert _normalize_tool_choice("required") == {"type": "required"}

    def test_unknown_string_defaults_auto(self):
        assert _normalize_tool_choice("bogus") == {"type": "auto"}

    def test_function_shape_to_tool(self):
        assert _normalize_tool_choice({"type": "function", "function": {"name": "my_tool"}}) == {
            "type": "tool", "toolName": "my_tool",
        }

    def test_v3_object_passthrough(self):
        assert _normalize_tool_choice({"type": "auto"}) == {"type": "auto"}
        assert _normalize_tool_choice({"type": "tool", "toolName": "x"}) == {"type": "tool", "toolName": "x"}

    def test_none_defaults_auto(self):
        assert _normalize_tool_choice(None) == {"type": "auto"}


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

    def test_assistant_tool_calls_are_content_parts(self):
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
        assert "toolCalls" not in assistant_msg
        assert assistant_msg["content"] == [
            {"type": "text", "text": ""},
            {"type": "tool-call", "toolCallId": "call_1", "toolName": "calc", "input": {"a": 1}},
        ]

    def test_assistant_tool_call_with_text_content(self):
        body = {
            "messages": [
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": "let me check",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }],
                },
            ],
        }
        result = openai_to_v3(body)
        parts = result["prompt"][1]["content"]
        assert parts[0] == {"type": "text", "text": "let me check"}
        assert parts[1]["type"] == "tool-call"

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

    def test_tool_result_without_prior_call_defaults_unknown(self):
        body = {"messages": [{"role": "tool", "tool_call_id": "x", "content": "y"}]}
        result = openai_to_v3(body)
        assert result["prompt"][0]["content"][0]["toolName"] == "unknown"

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

    def test_tool_choice_normalization_via_body(self):
        for raw, expected in [
            ("auto", {"type": "auto"}),
            ("none", {"type": "none"}),
            ("required", {"type": "required"}),
            ({"type": "function", "function": {"name": "t"}}, {"type": "tool", "toolName": "t"}),
        ]:
            body = {"messages": [{"role": "user", "content": "x"}], "tool_choice": raw}
            assert openai_to_v3(body)["toolChoice"] == expected

    def test_response_format_json_object(self):
        body = {"messages": [{"role": "user", "content": "x"}], "response_format": {"type": "json_object"}}
        assert openai_to_v3(body)["responseFormat"] == {
            "type": "json", "name": "", "description": "", "schema": {},
        }

    def test_response_format_json_schema(self):
        body = {"messages": [{"role": "user", "content": "x"}],
                "response_format": {"type": "json_schema", "json_schema": {
                    "name": "book", "description": "a book", "schema": {"type": "object"}}}}
        assert openai_to_v3(body)["responseFormat"] == {
            "type": "json", "name": "book", "description": "a book", "schema": {"type": "object"},
        }

    def test_response_format_absent(self):
        body = {"messages": [{"role": "user", "content": "x"}]}
        assert "responseFormat" not in openai_to_v3(body)

    def test_reasoning_passthrough(self):
        body = {"messages": [{"role": "user", "content": "x"}], "reasoning": {"effort": "high"}}
        assert openai_to_v3(body)["reasoning"] == {"effort": "high"}

    def test_reasoning_effort_mapping(self):
        body = {"messages": [{"role": "user", "content": "x"}], "reasoning_effort": "minimal"}
        assert openai_to_v3(body)["reasoning"] == "minimal"

    def test_provider_options_passthrough(self):
        body = {"messages": [{"role": "user", "content": "x"}],
                "providerOptions": {"gateway": {"cache": "auto"}}}
        assert openai_to_v3(body)["providerOptions"] == {"gateway": {"cache": "auto"}}

    def test_provider_options_empty_omitted(self):
        body = {"messages": [{"role": "user", "content": "x"}], "providerOptions": {}}
        assert "providerOptions" not in openai_to_v3(body)

    def test_input_body_not_mutated(self):
        import copy
        body = {"messages": [{"role": "user", "content": "x"}], "tool_choice": "auto"}
        snapshot = copy.deepcopy(body)
        openai_to_v3(body)
        assert body == snapshot


# =====================================================================
# validate_tool_history
# =====================================================================


class TestValidateToolHistory:
    def _call(self, call_id="call_1", name="calc", args="{}"):
        return {"id": call_id, "type": "function",
                "function": {"name": name, "arguments": args}}

    def test_valid_single_round(self):
        messages = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "tool_calls": [self._call()]},
            {"role": "tool", "tool_call_id": "call_1", "content": "42"},
            {"role": "user", "content": "thanks"},
        ]
        assert validate_tool_history(messages) is None

    def test_valid_multi_call_parallel(self):
        messages = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "tool_calls": [
                self._call("call_1", "calc"),
                self._call("call_2", "search"),
            ]},
            {"role": "tool", "tool_call_id": "call_2", "content": "b"},
            {"role": "tool", "tool_call_id": "call_1", "content": "a"},
        ]
        assert validate_tool_history(messages) is None

    def test_valid_two_rounds(self):
        messages = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "tool_calls": [self._call("call_1")]},
            {"role": "tool", "tool_call_id": "call_1", "content": "42"},
            {"role": "assistant", "content": None, "tool_calls": [self._call("call_2", "step2")]},
            {"role": "tool", "tool_call_id": "call_2", "content": "x"},
        ]
        assert validate_tool_history(messages) is None

    def test_no_tools_is_valid(self):
        assert validate_tool_history([{"role": "user", "content": "hi"}]) is None

    def test_orphan_tool_result_rejected(self):
        messages = [{"role": "user", "content": "go"}, {"role": "tool", "tool_call_id": "x", "content": "y"}]
        err = validate_tool_history(messages)
        assert err is not None and "no preceding assistant" in err

    def test_missing_results_rejected(self):
        messages = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "tool_calls": [self._call()]},
        ]
        err = validate_tool_history(messages)
        assert err is not None and "no matching tool results" in err

    def test_unmatched_result_id_rejected(self):
        messages = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "tool_calls": [self._call("call_1")]},
            {"role": "tool", "tool_call_id": "other", "content": "y"},
        ]
        err = validate_tool_history(messages)
        assert err is not None and "unknown tool call" in err

    def test_duplicate_tool_call_id_rejected(self):
        messages = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "tool_calls": [
                self._call("call_1"), self._call("call_1"),
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "a"},
            {"role": "tool", "tool_call_id": "call_1", "content": "b"},
        ]
        err = validate_tool_history(messages)
        assert err is not None and "duplicate tool call id" in err

    def test_invalid_json_args_rejected(self):
        messages = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "tool_calls": [
                self._call("call_1", args="not json"),
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "y"},
        ]
        err = validate_tool_history(messages)
        assert err is not None and "not valid JSON" in err

    def test_empty_arguments_rejected(self):
        messages = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "tool_calls": [
                self._call("call_1", args=""),
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "y"},
        ]
        err = validate_tool_history(messages)
        assert err is not None and "empty" in err

    def test_missing_tool_name_rejected(self):
        messages = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "tool_calls": [self._call(name="")]},
            {"role": "tool", "tool_call_id": "call_1", "content": "y"},
        ]
        err = validate_tool_history(messages)
        assert err is not None and "missing a function name" in err


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

    def test_tool_calls_from_top_level_toolCalls(self):
        v3 = {
            "content": [],
            "toolCalls": [{"toolCallId": "call_1", "toolName": "calc", "input": {"a": 1}}],
            "finishReason": "tool_calls",
        }
        result = v3_to_openai(v3, model="gpt-4")
        msg = result["choices"][0]["message"]
        assert msg["content"] is None
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["function"]["name"] == "calc"
        assert msg["tool_calls"][0]["id"] == "call_1"

    def test_tool_calls_from_openai_style(self):
        v3 = {
            "content": [],
            "tool_calls": [{
                "id": "call_9", "type": "function",
                "function": {"name": "f", "arguments": "{}"},
            }],
            "finishReason": "tool_calls",
        }
        result = v3_to_openai(v3)
        tc = result["choices"][0]["message"]["tool_calls"][0]
        assert tc["id"] == "call_9"
        assert tc["function"]["name"] == "f"

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

    def test_stream_finish_carries_usage(self):
        events = [
            {"type": "text-delta", "delta": "x"},
            {"type": "finish", "finishReason": "stop", "usage": {"inputTokens": {"total": 3}, "outputTokens": {"total": 2}}},
        ]
        sse = v3_stream_to_openai(events)
        finish_chunk = None
        for line in sse.split("\n\n"):
            if line.startswith("data: ") and "[DONE]" not in line:
                obj = json.loads(line[6:])
                if obj["choices"][0]["finish_reason"] == "stop":
                    finish_chunk = obj
        assert finish_chunk is not None
        assert finish_chunk["usage"] == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}

    def test_stream_no_usage_when_disabled(self):
        events = [
            {"type": "text-delta", "delta": "x"},
            {"type": "finish", "finishReason": "stop", "usage": {"inputTokens": {"total": 3}, "outputTokens": {"total": 2}}},
        ]
        sse = v3_stream_to_openai(events, include_usage=False)
        for line in sse.split("\n\n"):
            if line.startswith("data: ") and "[DONE]" not in line:
                obj = json.loads(line[6:])
                if obj["choices"][0]["finish_reason"] == "stop":
                    assert "usage" not in obj
                    return
        assert False, "finish chunk not found"

    def test_stream_multiple_tool_calls_sequential_indices(self):
        events = [
            {"type": "tool-call", "toolCallId": "call_1", "toolName": "read", "input": {}},
            {"type": "tool-call", "toolCallId": "call_2", "toolName": "write", "input": {}},
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
        delta1 = '{"a":'
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
                obj = json.loads(line[6:])
                tc = obj["choices"][0]["delta"].get("tool_calls", [])
                if tc and tc[0].get("function", {}).get("name") == "calc":
                    assert tc[0]["function"]["arguments"] == delta1 + delta2
                    return
        assert False, "tool-call chunk not found"

    def test_stream_async_iter_matches_sync(self):
        events = [
            {"type": "text-delta", "delta": "hi"},
            {"type": "finish", "finishReason": "stop"},
        ]
        sse = run_stream(events)
        assert "hi" in sse
        assert "data: [DONE]" in sse

    def test_stream_error_event(self):
        events = [
            {"type": "error", "error": {"message": "boom"}},
        ]
        sse = v3_stream_to_openai(events)
        assert '"type": "error"' in sse
        assert "boom" in sse
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
            {"type": "tool-call", "toolCallId": "call_1", "toolName": "run", "input": {}},
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


# =====================================================================
# Responses API translation
# =====================================================================


class TestResponsesInputToMessages:
    def test_flat_user_message(self):
        messages = responses_input_to_messages([{"role": "user", "content": "hi"}])
        assert messages == [{"role": "user", "content": "hi"}]

    def test_typed_message_item_with_text_blocks(self):
        messages = responses_input_to_messages([
            {"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "hello", "annotations": []},
            ]},
        ])
        assert messages == [{"role": "assistant", "content": "hello"}]

    def test_function_call_and_output(self):
        messages = responses_input_to_messages([
            {"type": "message", "role": "user", "content": "go"},
            {"type": "function_call", "call_id": "call_1", "name": "calc", "arguments": '{"a":1}'},
            {"type": "function_call_output", "call_id": "call_1", "output": "42"},
        ])
        assert messages == [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "calc", "arguments": '{"a":1}'},
                }],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "42"},
        ]

    def test_outputs_must_follow_their_call_batch(self):
        messages = responses_input_to_messages([
            {"type": "function_call", "call_id": "c1", "name": "a", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "1"},
            {"type": "function_call", "call_id": "c2", "name": "b", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c2", "output": "2"},
        ])
        # First call block + result, then second call block + result.
        assert [m["role"] for m in messages] == ["assistant", "tool", "assistant", "tool"]
        assert validate_tool_history(messages) is None

    def test_empty_input(self):
        assert responses_input_to_messages(None) == []
        assert responses_input_to_messages([]) == []


class TestOpenAIToResponses:
    def test_text_output(self):
        openai_resp = {
            "model": "gpt-4",
            "choices": [{"message": {"role": "assistant", "content": "hi"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        }
        result = openai_to_responses(openai_resp, model="gpt-4")
        assert result["object"] == "response"
        assert result["status"] == "completed"
        assert result["output"][0]["type"] == "message"
        assert result["output"][0]["content"] == [{"type": "output_text", "text": "hi", "annotations": []}]
        assert result["usage"]["total_tokens"] == 3

    def test_tool_call_output(self):
        openai_resp = {
            "choices": [{"message": {
                "role": "assistant", "content": None,
                "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "calc", "arguments": '{"a":1}'},
                }],
            }}],
        }
        result = openai_to_responses(openai_resp)
        assert result["output"][0]["type"] == "function_call"
        assert result["output"][0]["call_id"] == "call_1"
        assert result["output"][0]["name"] == "calc"
        assert result["output"][0]["status"] == "completed"

    def test_text_and_tool_both(self):
        openai_resp = {
            "choices": [{"message": {
                "role": "assistant", "content": "checking",
                "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "f", "arguments": "{}"},
                }],
            }}],
        }
        result = openai_to_responses(openai_resp)
        assert [o["type"] for o in result["output"]] == ["message", "function_call"]


class TestResponsesStreaming:
    def _chunk(self, **kw):
        return f"data: {json.dumps(kw)}\n\n"

    def test_stream_text(self):
        state = _ResponsesStreamState("gpt-4")
        chunk = self._chunk(
            id="x", object="chat.completion.chunk",
            choices=[{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        )
        out = openai_chunk_to_responses_sse(chunk, state)
        assert json.loads(out.split("\n\n")[0][6:])["type"] == "response.created"
        assert "response.output_item.added" in out
        assert "response.content_part.added" in out

        out = openai_chunk_to_responses_sse(
            self._chunk(choices=[{"index": 0, "delta": {"content": "hello"}, "finish_reason": None}]),
            state,
        )
        ev = json.loads(out.split("\n\n")[0][6:])
        assert ev["type"] == "response.output_text.delta"
        assert ev["delta"] == "hello"

        out = openai_chunk_to_responses_sse(
            self._chunk(choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}], usage={}),
            state,
        )
        assert "response.completed" in out
        final = json.loads([l for l in out.split("\n\n") if l.startswith("data:")]
                           [-1][6:])
        assert final["response"]["status"] == "completed"
        assert final["response"]["output"][0]["content"][0]["text"] == "hello"

    def test_stream_tool_call(self):
        state = _ResponsesStreamState("gpt-4")
        openai_chunk_to_responses_sse(
            self._chunk(choices=[{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]),
            state,
        )
        out = openai_chunk_to_responses_sse(
            self._chunk(choices=[{"index": 0, "delta": {"tool_calls": [{
                "index": 0, "id": "call_1", "type": "function",
                "function": {"name": "calc", "arguments": '{"a": 1}'},
            }]}, "finish_reason": None}]),
            state,
        )
        assert "response.output_item.added" in out
        assert "response.function_call_arguments.delta" in out

        out = openai_chunk_to_responses_sse(
            self._chunk(choices=[{"index": 0, "delta": {}, "finish_reason": "tool_calls"}], usage={}),
            state,
        )
        assert "response.completed" in out
        final = json.loads([l for l in out.split("\n\n") if l.startswith("data:")][-1][6:])
        fc = final["response"]["output"][0]
        assert fc["type"] == "function_call"
        assert fc["call_id"] == "call_1"
        assert fc["name"] == "calc"
        assert fc["status"] == "completed"

    def test_done_passthrough(self):
        state = _ResponsesStreamState("gpt-4")
        out = openai_chunk_to_responses_sse("data: [DONE]\n\n", state)
        assert out == "data: [DONE]\n\n"

    def test_error_event(self):
        state = _ResponsesStreamState("gpt-4")
        out = openai_chunk_to_responses_sse(
            self._chunk(type="error", error={"message": "boom"}), state)
        assert "response.failed" in out
        assert "boom" in out

    def test_offline_v3_to_responses(self):
        events = [
            {"type": "text-delta", "delta": "Hello"},
            {"type": "tool-call", "toolCallId": "call_1", "toolName": "calc", "input": {"a": 1}},
            {"type": "finish", "finishReason": "tool-calls",
             "usage": {"inputTokens": {"total": 5}, "outputTokens": {"total": 7}}},
        ]
        sse = v3_stream_to_responses_sse(events, model="gpt-4")
        assert "response.created" in sse
        assert "response.output_text.delta" in sse
        assert "response.function_call_arguments.delta" in sse
        assert "response.completed" in sse
        assert sse.rstrip().endswith("data: [DONE]")
        final = json.loads(
            [l for l in sse.split("\n\n")
             if l.startswith("data: ") and "[DONE]" not in l][-1][6:]
        )
        assert final["response"]["usage"]["total_tokens"] == 12
