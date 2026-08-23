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

    def test_created_timestamp_consistent_across_chunks(self):
        """All chunks in a stream must share the same `created` timestamp."""
        events = [
            {"type": "text-delta", "delta": "hel"},
            {"type": "text-delta", "delta": "lo"},
            {"type": "finish", "finishReason": "stop",
             "usage": {"inputTokens": {"total": 3}, "outputTokens": {"total": 2}}},
        ]
        sse = v3_stream_to_openai(events, model="gpt-4")
        created_values = set()
        for line in sse.split("\n\n"):
            if line.startswith("data: ") and "[DONE]" not in line:
                obj = json.loads(line[6:])
                created_values.add(obj["created"])
        assert len(created_values) == 1, f"expected 1 created value, got {created_values}"

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
# New fx streaming events (Task 6)
# =====================================================================


class TestStreamingNewEvents:
    def _data_chunks(self, sse):
        return [json.loads(l[6:]) for l in sse.split("\n\n") if l.startswith("data: ") and "[DONE]" not in l]

    def test_reasoning_delta_becomes_reasoning_content(self):
        events = [
            {"type": "reasoning-start", "id": "r1"},
            {"type": "reasoning-delta", "id": "r1", "delta": "let me think"},
            {"type": "reasoning-delta", "id": "r1", "delta": " harder"},
            {"type": "reasoning-end", "id": "r1"},
            {"type": "text-delta", "delta": "answer"},
            {"type": "finish", "finishReason": "stop"},
        ]
        sse = v3_stream_to_openai(events, model="gpt-4")
        reasoning = [c["choices"][0]["delta"].get("reasoning_content")
                     for c in self._data_chunks(sse)
                     if c["choices"][0]["delta"].get("reasoning_content")]
        assert reasoning == ["let me think", " harder"]

    def test_finish_step_usage_fallback(self):
        events = [
            {"type": "text-delta", "delta": "x"},
            {"type": "finish-step", "usage": {"inputTokens": {"total": 4}, "outputTokens": {"total": 2}}},
            {"type": "finish", "finishReason": "stop"},
        ]
        sse = v3_stream_to_openai(events)
        chunks = self._data_chunks(sse)
        finish = next(c for c in chunks if c["choices"][0]["finish_reason"] == "stop")
        assert finish["usage"]["total_tokens"] == 6

    def test_finish_usage_wins_over_step_usage(self):
        events = [
            {"type": "finish-step", "usage": {"inputTokens": {"total": 4}, "outputTokens": {"total": 2}}},
            {"type": "finish", "finishReason": "stop", "usage": {"inputTokens": {"total": 9}, "outputTokens": {"total": 1}}},
        ]
        sse = v3_stream_to_openai(events)
        chunks = self._data_chunks(sse)
        finish = next(c for c in chunks if c["choices"][0]["finish_reason"] == "stop")
        assert finish["usage"]["total_tokens"] == 10

    def test_extra_events_do_not_break_stream(self):
        events = [
            {"type": "start"},
            {"type": "start-step", "id": "s1"},
            {"type": "source", "source": {}},
            {"type": "file", "file": {}},
            {"type": "raw", "raw": {}},
            {"type": "tool-result", "toolCallId": "c1", "toolName": "t", "output": {"type": "text", "value": "ok"}},
            {"type": "text-delta", "delta": "hi"},
            {"type": "finish", "finishReason": "stop"},
        ]
        sse = v3_stream_to_openai(events, model="m")
        assert "hi" in sse
        assert sse.rstrip().endswith("data: [DONE]")

    def test_unknown_event_ignored(self):
        events = [
            {"type": "mystery-event", "payload": {}},
            {"type": "text-delta", "delta": "ok"},
            {"type": "finish", "finishReason": "stop"},
        ]
        sse = v3_stream_to_openai(events, model="m")
        assert "ok" in sse
        assert sse.rstrip().endswith("data: [DONE]")

    def test_response_metadata_sets_model_when_empty(self):
        events = [
            {"type": "response-metadata", "id": "x", "modelId": "glm52"},
            {"type": "text-delta", "delta": "hi"},
            {"type": "finish", "finishReason": "stop"},
        ]
        sse = v3_stream_to_openai(events, model="")
        chunks = self._data_chunks(sse)
        assert all(c["model"] == "glm52" for c in chunks)

    def test_non_stream_collector_uses_step_usage_fallback(self):
        events = [
            {"type": "text-delta", "delta": "x"},
            {"type": "finish-step", "usage": {"inputTokens": {"total": 4}, "outputTokens": {"total": 2}}},
            {"type": "finish", "finishReason": "stop"},
        ]
        result = v3_sse_stream_to_openai(iter(events), model="m")
        assert result["usage"] == {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}

    def test_non_stream_collector_collects_reasoning(self):
        events = [
            {"type": "reasoning-delta", "delta": "let me think"},
            {"type": "reasoning-delta", "delta": " harder"},
            {"type": "text-delta", "delta": "answer"},
            {"type": "finish", "finishReason": "stop"},
        ]
        result = v3_sse_stream_to_openai(iter(events), model="m")
        msg = result["choices"][0]["message"]
        assert msg["reasoning_content"] == "let me think harder"
        assert msg["content"] == "answer"

    def test_non_stream_collector_no_reasoning_omits_field(self):
        events = [
            {"type": "text-delta", "delta": "answer"},
            {"type": "finish", "finishReason": "stop"},
        ]
        result = v3_sse_stream_to_openai(iter(events), model="m")
        assert "reasoning_content" not in result["choices"][0]["message"]

    def test_non_stream_collector_reasoning_with_tools(self):
        events = [
            {"type": "reasoning-delta", "delta": "thinking about tools"},
            {"type": "tool-call", "toolCallId": "call_1", "toolName": "calc", "input": {"x": 1}},
            {"type": "finish", "finishReason": "tool-calls",
             "usage": {"inputTokens": {"total": 5}, "outputTokens": {"total": 3}}},
        ]
        result = v3_sse_stream_to_openai(iter(events), model="m")
        msg = result["choices"][0]["message"]
        assert msg["reasoning_content"] == "thinking about tools"
        assert msg["content"] is None
        assert len(msg["tool_calls"]) == 1
        assert result["choices"][0]["finish_reason"] == "tool_calls"

    def test_non_stream_collector_reasoning_with_usage(self):
        events = [
            {"type": "reasoning-delta", "delta": "thinking"},
            {"type": "text-delta", "delta": "answer"},
            {"type": "finish", "finishReason": "stop",
             "usage": {"inputTokens": {"total": 10}, "outputTokens": {"total": 5, "reasoning": 3}}},
        ]
        result = v3_sse_stream_to_openai(iter(events), model="m")
        msg = result["choices"][0]["message"]
        assert msg["reasoning_content"] == "thinking"
        assert msg["content"] == "answer"
        assert result["usage"]["output_tokens_details"] == {"reasoning_tokens": 3}

    def test_stream_reasoning_then_text_then_tool(self):
        """Full stream: reasoning deltas, text deltas, tool call, finish."""
        events = [
            {"type": "reasoning-start", "id": "r1"},
            {"type": "reasoning-delta", "id": "r1", "delta": "let me think"},
            {"type": "reasoning-end", "id": "r1"},
            {"type": "text-delta", "delta": "here is "},
            {"type": "text-delta", "delta": "my answer"},
            {"type": "tool-call", "toolCallId": "call_1", "toolName": "save", "input": {"result": "done"}},
            {"type": "finish", "finishReason": "tool-calls",
             "usage": {"inputTokens": {"total": 10}, "outputTokens": {"total": 5}}},
        ]
        sse = v3_stream_to_openai(events, model="gpt-4")
        chunks = self._data_chunks(sse)
        reasoning = [c["choices"][0]["delta"].get("reasoning_content")
                     for c in chunks
                     if c["choices"][0]["delta"].get("reasoning_content")]
        assert reasoning == ["let me think"]
        text_parts = [c["choices"][0]["delta"].get("content")
                      for c in chunks
                      if c["choices"][0]["delta"].get("content")]
        assert text_parts == ["here is ", "my answer"]
        tool_chunks = [c for c in chunks if c["choices"][0]["delta"].get("tool_calls")]
        assert len(tool_chunks) == 1
        assert tool_chunks[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "save"

    def test_stream_reasoning_only_no_text(self):
        events = [
            {"type": "reasoning-delta", "delta": "just thinking"},
            {"type": "finish", "finishReason": "stop"},
        ]
        sse = v3_stream_to_openai(events, model="gpt-4")
        chunks = self._data_chunks(sse)
        reasoning = [c["choices"][0]["delta"].get("reasoning_content")
                     for c in chunks
                     if c["choices"][0]["delta"].get("reasoning_content")]
        assert reasoning == ["just thinking"]
        # no text content chunks
        text = [c for c in chunks if c["choices"][0]["delta"].get("content")]
        assert text == []
