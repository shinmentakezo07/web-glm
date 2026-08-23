"""Unit tests for converter.responses — Responses API <-> v3 translation."""

import json

from converter.responses import (
    _ResponsesStreamState,
    openai_chunk_to_responses_sse,
    openai_to_responses,
    responses_input_to_messages,
    v3_stream_to_responses_sse,
)
from converter.validation import validate_tool_history


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

        out = openai_chunk_to_responses_sse(
            self._chunk(choices=[{"index": 0, "delta": {"content": "hello"}, "finish_reason": None}]),
            state,
        )
        # Message item is created lazily on first text delta.
        assert "response.output_item.added" in out
        assert "response.content_part.added" in out
        delta_events = [e for e in out.split("\n\n") if "output_text.delta" in e]
        assert delta_events
        ev = json.loads(delta_events[0][6:])
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
        # Tool-only response: no empty message item in the output.
        fc = final["response"]["output"][0]
        assert fc["type"] == "function_call"
        assert fc["call_id"] == "call_1"
        assert fc["name"] == "calc"
        assert fc["status"] == "completed"
        assert len(final["response"]["output"]) == 1, "no dangling empty message"

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
