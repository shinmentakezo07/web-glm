"""Unit tests for converter.streaming — v3 SSE stream -> OpenAI SSE chunks."""

import asyncio
import json

from converter.streaming import (
    v3_sse_stream_to_openai,
    v3_stream_iter,
    v3_stream_to_openai,
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
