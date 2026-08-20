"""Unit tests for converter.response — v3 gateway response -> OpenAI."""

from converter.response import _v3_finish_reason, _v3_usage_to_openai, v3_to_openai


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
