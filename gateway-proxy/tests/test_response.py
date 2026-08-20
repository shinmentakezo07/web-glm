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


class TestUsageDetails:
    def test_cached_tokens_mapped(self):
        usage = _v3_usage_to_openai({
            "inputTokens": {"total": 100, "cacheRead": 80},
            "outputTokens": {"total": 10},
        })
        assert usage["prompt_tokens"] == 100
        assert usage["prompt_tokens_details"] == {"cached_tokens": 80}

    def test_reasoning_tokens_mapped(self):
        usage = _v3_usage_to_openai({
            "inputTokens": {"total": 5},
            "outputTokens": {"total": 20, "reasoning": 15},
        })
        assert usage["completion_tokens"] == 20
        assert usage["output_tokens_details"] == {"reasoning_tokens": 15}

    def test_details_omitted_when_absent(self):
        usage = _v3_usage_to_openai({
            "inputTokens": {"total": 5},
            "outputTokens": {"total": 10},
        })
        assert usage == {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}
        assert "prompt_tokens_details" not in usage
        assert "output_tokens_details" not in usage

    def test_flat_input_output_tokens(self):
        usage = _v3_usage_to_openai({"inputTokens": 7, "outputTokens": 3})
        assert usage == {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
