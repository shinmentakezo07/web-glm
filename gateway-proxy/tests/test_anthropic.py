"""Unit tests for converter.anthropic — Anthropic Messages API <-> OpenAI/v3."""

import asyncio
import json

from converter.anthropic import (
    _AnthropicStreamState,
    anthropic_stream_iter,
    anthropic_to_openai,
    count_anthropic_tokens,
    openai_chunk_to_anthropic_sse,
    openai_to_anthropic,
)


# =====================================================================
# Anthropic -> OpenAI request conversion
# =====================================================================


class TestAnthropicToOpenAI:
    def test_simple_user_message(self):
        body = {
            "model": "claude-3",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hello"}],
        }
        result = anthropic_to_openai(body)
        assert result["messages"] == [{"role": "user", "content": "hello"}]
        assert result["model"] == "claude-3"
        assert result["max_tokens"] == 100

    def test_system_string_becomes_system_message(self):
        body = {
            "max_tokens": 10,
            "system": "you are helpful",
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = anthropic_to_openai(body)
        assert result["messages"][0] == {"role": "system", "content": "you are helpful"}
        assert result["messages"][1] == {"role": "user", "content": "hi"}

    def test_system_list_of_blocks_concatenated(self):
        body = {
            "max_tokens": 10,
            "system": [
                {"type": "text", "text": "rule 1. "},
                {"type": "text", "text": "rule 2."},
            ],
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = anthropic_to_openai(body)
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["content"] == "rule 1. rule 2."

    def test_assistant_tool_use_becomes_tool_calls(self):
        body = {
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "run it"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "sure"},
                    {"type": "tool_use", "id": "toolu_1", "name": "calc", "input": {"x": 1}},
                ]},
            ],
        }
        result = anthropic_to_openai(body)
        assistant = result["messages"][1]
        assert assistant["role"] == "assistant"
        assert assistant["content"] == "sure"
        assert assistant["tool_calls"][0]["id"] == "toolu_1"
        assert assistant["tool_calls"][0]["function"]["name"] == "calc"
        assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"x": 1}

    def test_tool_result_becomes_tool_message(self):
        body = {
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "run it"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "calc", "input": {"x": 1}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "42"},
                ]},
            ],
        }
        result = anthropic_to_openai(body)
        # user -> assistant(tool_use) -> tool(result)
        assert [m["role"] for m in result["messages"]] == ["user", "assistant", "tool"]
        tool_msg = result["messages"][2]
        assert tool_msg["tool_call_id"] == "toolu_1"
        assert tool_msg["content"] == "42"

    def test_tool_result_with_list_content(self):
        body = {
            "max_tokens": 100,
            "messages": [
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "f", "input": {}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": [
                        {"type": "text", "text": "result: "},
                        {"type": "text", "text": "ok"},
                    ]},
                ]},
            ],
        }
        result = anthropic_to_openai(body)
        tool_msg = result["messages"][1]
        assert tool_msg["content"] == "result: ok"

    def test_image_block_becomes_image_url(self):
        body = {
            "max_tokens": 10,
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "iVBOR",
                }},
                {"type": "text", "text": "what is this?"},
            ]}],
        }
        result = anthropic_to_openai(body)
        msg = result["messages"][0]
        assert isinstance(msg["content"], list)
        assert msg["content"][0]["type"] == "image_url"
        assert "data:image/png;base64,iVBOR" in msg["content"][0]["image_url"]["url"]
        assert msg["content"][1] == {"type": "text", "text": "what is this?"}

    def test_tools_converted_to_openai_function_format(self):
        body = {
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
            }],
        }
        result = anthropic_to_openai(body)
        assert result["tools"][0] == {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }

    def test_tool_choice_auto(self):
        body = {"max_tokens": 10, "messages": [{"role": "user", "content": "x"}],
                "tool_choice": {"type": "auto"}}
        assert anthropic_to_openai(body)["tool_choice"] == "auto"

    def test_tool_choice_any_becomes_required(self):
        body = {"max_tokens": 10, "messages": [{"role": "user", "content": "x"}],
                "tool_choice": {"type": "any"}}
        assert anthropic_to_openai(body)["tool_choice"] == "required"

    def test_tool_choice_tool_becomes_function(self):
        body = {"max_tokens": 10, "messages": [{"role": "user", "content": "x"}],
                "tool_choice": {"type": "tool", "name": "get_weather"}}
        result = anthropic_to_openai(body)
        assert result["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}

    def test_params_passthrough(self):
        body = {
            "max_tokens": 100,
            "temperature": 0.5,
            "top_p": 0.9,
            "top_k": 40,
            "stop_sequences": ["END"],
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = anthropic_to_openai(body)
        assert result["max_tokens"] == 100
        assert result["temperature"] == 0.5
        assert result["top_p"] == 0.9
        assert result["top_k"] == 40
        assert result["stop"] == ["END"]

    def test_stream_flag_passthrough(self):
        body = {"max_tokens": 10, "stream": True, "messages": [{"role": "user", "content": "x"}]}
        assert anthropic_to_openai(body)["stream"] is True

    def test_thinking_enabled_routes_reasoning(self):
        body = {
            "max_tokens": 100,
            "thinking": {"type": "enabled", "budget_tokens": 10000},
            "messages": [{"role": "user", "content": "think hard"}],
        }
        result = anthropic_to_openai(body)
        assert result["reasoning"] == "enabled"

    def test_thinking_disabled_no_reasoning(self):
        body = {
            "max_tokens": 100,
            "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": "no thinking"}],
        }
        result = anthropic_to_openai(body)
        assert "reasoning" not in result

    def test_no_thinking_no_reasoning(self):
        body = {"max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]}
        result = anthropic_to_openai(body)
        assert "reasoning" not in result

    def test_explicit_reasoning_effort_overrides_thinking(self):
        body = {
            "max_tokens": 100,
            "thinking": {"type": "enabled", "budget_tokens": 5000},
            "reasoning_effort": "high",
            "messages": [{"role": "user", "content": "x"}],
        }
        result = anthropic_to_openai(body)
        # reasoning_effort is set, so thinking doesn't override it
        assert "reasoning" not in result or result.get("reasoning") != "enabled"
        assert result["reasoning_effort"] == "high"

    def test_explicit_reasoning_string_overrides_thinking(self):
        body = {
            "max_tokens": 100,
            "thinking": {"type": "enabled", "budget_tokens": 5000},
            "reasoning": "max",
            "messages": [{"role": "user", "content": "x"}],
        }
        result = anthropic_to_openai(body)
        assert result["reasoning"] == "max"

    def test_input_body_not_mutated(self):
        import copy
        body = {
            "max_tokens": 100,
            "system": "sys",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "tools": [{"name": "f", "input_schema": {}}],
        }
        snapshot = copy.deepcopy(body)
        anthropic_to_openai(body)
        assert body == snapshot


# =====================================================================
# OpenAI -> Anthropic response conversion (non-streaming)
# =====================================================================


class TestOpenAIToAnthropic:
    def test_text_response(self):
        openai_resp = {
            "id": "chatcmpl-123",
            "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }
        result = openai_to_anthropic(openai_resp, model="claude-3")
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["content"] == [{"type": "text", "text": "hi"}]
        assert result["stop_reason"] == "end_turn"
        assert result["usage"]["input_tokens"] == 3
        assert result["usage"]["output_tokens"] == 2

    def test_tool_use_response(self):
        openai_resp = {
            "id": "chatcmpl-123",
            "choices": [{"message": {
                "role": "assistant", "content": None,
                "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "calc", "arguments": '{"x":1}'},
                }],
            }, "finish_reason": "tool_calls"}],
        }
        result = openai_to_anthropic(openai_resp, model="claude-3")
        assert result["stop_reason"] == "tool_use"
        assert result["content"][0]["type"] == "tool_use"
        assert result["content"][0]["name"] == "calc"
        assert result["content"][0]["input"] == {"x": 1}

    def test_text_and_tool_combined(self):
        openai_resp = {
            "choices": [{"message": {
                "role": "assistant", "content": "checking",
                "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "f", "arguments": "{}"},
                }],
            }, "finish_reason": "tool_calls"}],
        }
        result = openai_to_anthropic(openai_resp, model="claude-3")
        types = [c["type"] for c in result["content"]]
        assert types == ["text", "tool_use"]

    def test_reasoning_becomes_thinking_block(self):
        openai_resp = {
            "choices": [{"message": {
                "role": "assistant",
                "content": "the answer",
                "reasoning_content": "I reasoned about this",
            }, "finish_reason": "stop"}],
        }
        result = openai_to_anthropic(openai_resp, model="claude-3")
        types = [c["type"] for c in result["content"]]
        assert types == ["thinking", "text"]
        assert result["content"][0]["thinking"] == "I reasoned about this"
        assert result["content"][1]["text"] == "the answer"

    def test_reasoning_only_no_text(self):
        openai_resp = {
            "choices": [{"message": {
                "role": "assistant", "content": None,
                "reasoning_content": "just thinking",
            }, "finish_reason": "stop"}],
        }
        result = openai_to_anthropic(openai_resp, model="claude-3")
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "thinking"

    def test_reasoning_and_tool_combined(self):
        openai_resp = {
            "choices": [{"message": {
                "role": "assistant",
                "content": "checking",
                "reasoning_content": "I need to use a tool",
                "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }],
            }, "finish_reason": "tool_calls"}],
        }
        result = openai_to_anthropic(openai_resp, model="claude-3")
        types = [c["type"] for c in result["content"]]
        assert types == ["thinking", "text", "tool_use"]
        assert result["content"][0]["thinking"] == "I need to use a tool"
        assert result["content"][1]["text"] == "checking"
        assert result["content"][2]["name"] == "lookup"
        assert result["stop_reason"] == "tool_use"

    def test_reasoning_tokens_in_usage(self):
        openai_resp = {
            "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 10, "completion_tokens": 20,
                "completion_tokens_details": {"reasoning_tokens": 15},
            },
        }
        result = openai_to_anthropic(openai_resp)
        # Anthropic doesn't have a direct reasoning_tokens field, but
        # output_tokens should still be the total
        assert result["usage"]["output_tokens"] == 20

    def test_empty_content_becomes_empty_text(self):
        openai_resp = {"choices": [{"message": {"role": "assistant", "content": None}, "finish_reason": "stop"}]}
        result = openai_to_anthropic(openai_resp, model="claude-3")
        assert result["content"] == [{"type": "text", "text": ""}]

    def test_finish_reason_mappings(self):
        assert openai_to_anthropic({"choices": [{"message": {}, "finish_reason": "stop"}]})["stop_reason"] == "end_turn"
        assert openai_to_anthropic({"choices": [{"message": {}, "finish_reason": "length"}]})["stop_reason"] == "max_tokens"
        assert openai_to_anthropic({"choices": [{"message": {}, "finish_reason": "tool_calls"}]})["stop_reason"] == "tool_use"

    def test_usage_with_cache_read(self):
        openai_resp = {
            "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 10, "completion_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": 3},
            },
        }
        result = openai_to_anthropic(openai_resp)
        assert result["usage"]["cache_read_input_tokens"] == 3
        assert result["usage"]["input_tokens"] == 10


# =====================================================================
# Streaming conversion
# =====================================================================


class TestAnthropicStreaming:
    def _chunk(self, **kw):
        return f"data: {json.dumps(kw)}\n\n"

    def _parse_events(self, sse: str) -> list[dict]:
        events = []
        for block in sse.split("\n\n"):
            block = block.strip()
            if not block:
                continue
            for line in block.split("\n"):
                if line.startswith("data: "):
                    try:
                        events.append(json.loads(line[6:]))
                    except json.JSONDecodeError:
                        pass
        return events

    def test_stream_text(self):
        state = _AnthropicStreamState("claude-3")
        out = openai_chunk_to_anthropic_sse(self._chunk(
            id="x", object="chat.completion.chunk",
            choices=[{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        ), state)
        assert "message_start" in out

        out = openai_chunk_to_anthropic_sse(self._chunk(
            choices=[{"index": 0, "delta": {"content": "hello"}, "finish_reason": None}],
        ), state)
        assert "content_block_start" in out
        assert "content_block_delta" in out
        assert "text_delta" in out

        out = openai_chunk_to_anthropic_sse(self._chunk(
            choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}],
            usage={"prompt_tokens": 1, "completion_tokens": 1},
        ), state)
        assert "content_block_stop" in out
        assert "message_delta" in out
        assert "message_stop" in out

        events = self._parse_events(out)
        delta = next(e for e in events if e["type"] == "message_delta")
        assert delta["delta"]["stop_reason"] == "end_turn"

    def test_stream_tool_call(self):
        state = _AnthropicStreamState("claude-3")
        openai_chunk_to_anthropic_sse(self._chunk(
            choices=[{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        ), state)
        out = openai_chunk_to_anthropic_sse(self._chunk(
            choices=[{"index": 0, "delta": {"tool_calls": [{
                "index": 0, "id": "call_1", "type": "function",
                "function": {"name": "calc", "arguments": '{"x":'},
            }]}, "finish_reason": None}],
        ), state)
        assert "content_block_start" in out
        assert "tool_use" in out
        assert "input_json_delta" in out

        out = openai_chunk_to_anthropic_sse(self._chunk(
            choices=[{"index": 0, "delta": {"tool_calls": [{
                "index": 0, "function": {"arguments": "1}"},
            }]}, "finish_reason": None}],
        ), state)
        assert "input_json_delta" in out

        out = openai_chunk_to_anthropic_sse(self._chunk(
            choices=[{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            usage={"prompt_tokens": 1, "completion_tokens": 1},
        ), state)
        assert "content_block_stop" in out
        assert "message_stop" in out

        events = self._parse_events(out)
        delta = next(e for e in events if e["type"] == "message_delta")
        assert delta["delta"]["stop_reason"] == "tool_use"

    def test_stream_error(self):
        state = _AnthropicStreamState("claude-3")
        out = openai_chunk_to_anthropic_sse(
            self._chunk(type="error", error={"message": "boom"}), state
        )
        assert "error" in out
        assert "boom" in out

    def test_stream_done_passthrough_none(self):
        state = _AnthropicStreamState("claude-3")
        result = openai_chunk_to_anthropic_sse("data: [DONE]\n\n", state)
        assert result is None

    def test_text_then_tool_indices(self):
        """Text block at index 0, tool block at index 1."""
        state = _AnthropicStreamState("claude-3")
        openai_chunk_to_anthropic_sse(self._chunk(
            choices=[{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        ), state)
        # Text
        openai_chunk_to_anthropic_sse(self._chunk(
            choices=[{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}],
        ), state)
        # Tool call
        out = openai_chunk_to_anthropic_sse(self._chunk(
            choices=[{"index": 0, "delta": {"tool_calls": [{
                "index": 0, "id": "call_1", "type": "function",
                "function": {"name": "f", "arguments": "{}"},
            }]}, "finish_reason": None}],
        ), state)
        # SSE blocks are "event: ...\ndata: {...}\n\n" — split and parse data lines
        start_events = [json.loads(l[6:]) for l in out.split("\n")
                        if l.strip().startswith("data:")
                        and "content_block_start" in l]
        assert len(start_events) == 1
        assert start_events[0]["index"] == 1  # tool block at index 1

    def test_stream_thinking_block(self):
        """Reasoning content becomes a thinking block, not text."""
        state = _AnthropicStreamState("claude-3")
        openai_chunk_to_anthropic_sse(self._chunk(
            choices=[{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        ), state)
        # Reasoning delta
        out = openai_chunk_to_anthropic_sse(self._chunk(
            choices=[{"index": 0, "delta": {"reasoning_content": "thinking..."}, "finish_reason": None}],
        ), state)
        assert "content_block_start" in out
        assert "thinking" in out
        assert "thinking_delta" in out

        # Text delta — should be a separate block; closes thinking first
        out = openai_chunk_to_anthropic_sse(self._chunk(
            choices=[{"index": 0, "delta": {"content": "answer"}, "finish_reason": None}],
        ), state)
        assert "text_delta" in out
        assert "content_block_start" in out  # new text block started
        assert "content_block_stop" in out  # thinking block closed

        # Finish — should close the text block (thinking already closed)
        out = openai_chunk_to_anthropic_sse(self._chunk(
            choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}],
            usage={"prompt_tokens": 1, "completion_tokens": 1},
        ), state)
        # One content_block_stop event for the text block (thinking closed earlier)
        stops = [l for l in out.split("\n") if l.strip().startswith("data:")
                 and "content_block_stop" in l]
        assert len(stops) == 1
        assert "message_stop" in out

    def test_stream_thinking_then_tool_indices(self):
        """Thinking at index 0, text at index 1, tool at index 2."""
        state = _AnthropicStreamState("claude-3")
        openai_chunk_to_anthropic_sse(self._chunk(
            choices=[{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        ), state)
        # Thinking
        out_thinking = openai_chunk_to_anthropic_sse(self._chunk(
            choices=[{"index": 0, "delta": {"reasoning_content": "hm"}, "finish_reason": None}],
        ), state)
        thinking_start = [json.loads(l[6:]) for l in out_thinking.split("\n")
                          if l.strip().startswith("data:")
                          and "content_block_start" in l]
        assert thinking_start[0]["index"] == 0
        assert thinking_start[0]["content_block"]["type"] == "thinking"

        # Text
        out_text = openai_chunk_to_anthropic_sse(self._chunk(
            choices=[{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}],
        ), state)
        text_start = [json.loads(l[6:]) for l in out_text.split("\n")
                      if l.strip().startswith("data:")
                      and "content_block_start" in l]
        assert text_start[0]["index"] == 1
        assert text_start[0]["content_block"]["type"] == "text"

        # Tool
        out_tool = openai_chunk_to_anthropic_sse(self._chunk(
            choices=[{"index": 0, "delta": {"tool_calls": [{
                "index": 0, "id": "call_1", "type": "function",
                "function": {"name": "f", "arguments": "{}"},
            }]}, "finish_reason": None}],
        ), state)
        tool_start = [json.loads(l[6:]) for l in out_tool.split("\n")
                      if l.strip().startswith("data:")
                      and "content_block_start" in l]
        assert tool_start[0]["index"] == 2
        assert tool_start[0]["content_block"]["type"] == "tool_use"


# =====================================================================
# Async streaming
# =====================================================================


def _async_iter(chunks):
    async def gen():
        for c in chunks:
            yield c
    return gen()


def run_anthropic_stream(chunks, model="claude-3"):
    async def _collect():
        result = []
        async for c in anthropic_stream_iter(_async_iter(chunks), model):
            result.append(c)
        return "".join(result)
    return asyncio.run(_collect())


class TestAnthropicStreamAsync:
    def test_full_stream_text(self):
        chunks = [
            f"data: {json.dumps({'id': 'x', 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n",
            f"data: {json.dumps({'id': 'x', 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {'content': 'hi'}, 'finish_reason': None}]})}\n\n",
            f"data: {json.dumps({'id': 'x', 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 1, 'completion_tokens': 1}})}\n\n",
        ]
        sse = run_anthropic_stream(chunks)
        assert "message_start" in sse
        assert "content_block_start" in sse
        assert "text_delta" in sse
        assert "content_block_stop" in sse
        assert "message_delta" in sse
        assert "message_stop" in sse

    def test_empty_stream_emits_start_and_stop(self):
        sse = run_anthropic_stream([])
        assert "message_start" in sse
        assert "message_stop" in sse

    def test_full_stream_with_reasoning(self):
        chunks = [
            f"data: {json.dumps({'id': 'x', 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n",
            f"data: {json.dumps({'id': 'x', 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {'reasoning_content': 'thinking...'}, 'finish_reason': None}]})}\n\n",
            f"data: {json.dumps({'id': 'x', 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {'content': 'answer'}, 'finish_reason': None}]})}\n\n",
            f"data: {json.dumps({'id': 'x', 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 1, 'completion_tokens': 1}})}\n\n",
        ]
        sse = run_anthropic_stream(chunks)
        assert "message_start" in sse
        assert "thinking_delta" in sse
        assert "text_delta" in sse
        # thinking block should be closed before text block starts
        thinking_stop_idx = sse.index('"type": "content_block_stop"')
        text_start_idx = sse.index('"type": "text_delta"')
        assert thinking_stop_idx < text_start_idx
        assert "message_stop" in sse

    def test_full_stream_reasoning_then_tool(self):
        chunk_data = [
            {"id": "x", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
            {"id": "x", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"reasoning_content": "need a tool"}, "finish_reason": None}]},
            {"id": "x", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}], "finish_reason": None}}]},
            {"id": "x", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
        ]
        chunks = [f"data: {json.dumps(d)}\n\n" for d in chunk_data]
        sse = run_anthropic_stream(chunks)
        assert "thinking_delta" in sse
        assert "input_json_delta" in sse
        assert "tool_use" in sse
        assert "message_stop" in sse


# =====================================================================
# Token counting
# =====================================================================


class TestCountTokens:
    def test_simple_text(self):
        body = {"messages": [{"role": "user", "content": "hello world"}]}
        count = count_anthropic_tokens(body)
        assert count == 3  # ceil(11/4) = 3

    def test_system_counted(self):
        body = {
            "system": "you are a bot",
            "messages": [{"role": "user", "content": "hi"}],
        }
        count = count_anthropic_tokens(body)
        assert count > 0

    def test_tools_counted(self):
        body = {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{
                "name": "f",
                "description": "a tool",
                "input_schema": {"type": "object", "properties": {"x": {"type": "string"}}},
            }],
        }
        count = count_anthropic_tokens(body)
        assert count > 2  # tools add chars

    def test_empty_body(self):
        assert count_anthropic_tokens({}) == 0

    def test_tool_result_content_counted(self):
        body = {
            "messages": [
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "x", "content": "a long result string"},
                ]},
            ],
        }
        count = count_anthropic_tokens(body)
        assert count > 0
