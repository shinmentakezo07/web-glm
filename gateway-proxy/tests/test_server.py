"""Server route tests with a mocked upstream gateway.

A deterministic httpx.MockTransport is injected into app.state.client so no
real network traffic is needed. The mock records the v3 request bodies, which
lets us assert the wire format the proxy actually sends upstream.
"""

import json
import os

os.environ["PROXY_API_KEY"] = "test-proxy-key"
os.environ["AI_GATEWAY_API_KEY"] = "test-gateway-key"
os.environ["GATEWAY_HTTP2"] = "0"
os.environ["MODELS_CACHE_TTL"] = "300"

import httpx
from fastapi.testclient import TestClient

import server


def sse_chat(v3_request_body: dict) -> httpx.Response:
    return httpx.Response(
        200,
        content=(
            'data: {"type":"text-delta","delta":"hi"}\n\n'
            'data: {"type":"text-delta","delta":" there"}\n\n'
            'data: {"type":"finish","finishReason":"stop",'
            '"usage":{"inputTokens":{"total":3},"outputTokens":{"total":2}}}\n\n'
            "data: [DONE]\n\n"
        ),
        headers={"content-type": "text/event-stream"},
    )



def sse_reasoning(v3_request_body: dict) -> httpx.Response:
    """Upstream response with reasoning-delta events before text."""
    return httpx.Response(
        200,
        content=(
            'data: {"type":"reasoning-start","id":"r1"}\n\n'
            'data: {"type":"reasoning-delta","id":"r1","delta":"let me think"}\n\n'
            'data: {"type":"reasoning-delta","id":"r1","delta":" harder"}\n\n'
            'data: {"type":"reasoning-end","id":"r1"}\n\n'
            'data: {"type":"text-delta","delta":"the answer"}\n\n'
            'data: {"type":"finish","finishReason":"stop",'
            '"usage":{"inputTokens":{"total":5},"outputTokens":{"total":3,"reasoning":2}}}\n\n'
            "data: [DONE]\n\n"
        ),
        headers={"content-type": "text/event-stream"},
    )

def sse_tool_call(v3_request_body: dict) -> httpx.Response:
    return httpx.Response(
        200,
        content=(
            'data: {"type":"tool-call","toolCallId":"call_1","toolName":"calc","input":{"a":1}}\n\n'
            'data: {"type":"finish","finishReason":"tool-calls",'
            '"usage":{"inputTokens":{"total":4},"outputTokens":{"total":1}}}\n\n'
            "data: [DONE]\n\n"
        ),
        headers={"content-type": "text/event-stream"},
    )


def sse_error(v3_request_body: dict) -> httpx.Response:
    return httpx.Response(400, json={
        "error": {"message": "Invalid input", "path": ["toolChoice"]},
    })


def make_router(calls: list[dict]) -> httpx.MockTransport:
    """Transport that records calls and serves canned responses per path."""
    models_hits = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append({
            "method": request.method,
            "url": str(request.url),
            "headers": dict(request.headers),
            "body": json.loads(request.content) if request.content else None,
        })
        path = request.url.path
        if path == "/v1/models":
            models_hits["count"] += 1
            return httpx.Response(200, json={"data": [{"id": "m1"}, {"id": "m2"}]})
        if path == "/v3/ai/language-model":
            body = json.loads(request.content) if request.content else {}
            prompt_text = json.dumps(body.get("prompt", []))
            # Check for reasoning request: v3 body has "reasoning" field
            if body.get("reasoning"):
                return sse_reasoning(body)
            if any(
                isinstance(m, dict) and m.get("type") == "tool-call"
                for msg in body.get("prompt", [])
                for m in (msg.get("content") or [])
                if isinstance(msg, dict)
            ):
                return sse_tool_call(body)
            if "Invalid input" in prompt_text:
                return sse_error(body)
            return sse_chat(body)
        if path == "/v1/embeddings":
            return httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2]}]})
        return httpx.Response(404, json={"error": {"message": "not found"}})

    return httpx.MockTransport(handler)


def setup_test_client(calls: list[dict]) -> TestClient:
    server.app.state.client = httpx.AsyncClient(
        transport=make_router(calls), timeout=httpx.Timeout(5.0)
    )
    server.app.state.models_cache = {"data": None, "expires": 0.0}
    return TestClient(server.app)


AUTH_HEADERS = {"Authorization": "Bearer test-proxy-key"}


class TestAuth:
    def test_no_key_rejected(self):
        with setup_test_client([]) as client:
            resp = client.get("/v1/models")
            assert resp.status_code == 401

    def test_wrong_key_rejected(self):
        with setup_test_client([]) as client:
            resp = client.get("/v1/models", headers={"Authorization": "Bearer nope"})
            assert resp.status_code == 401

    def test_valid_key_accepted(self):
        with setup_test_client([]) as client:
            resp = client.get("/v1/models", headers=AUTH_HEADERS)
            assert resp.status_code == 200


class TestModels:
    def test_caching(self):
        calls: list[dict] = []
        with setup_test_client(calls) as client:
            first = client.get("/v1/models", headers=AUTH_HEADERS)
            second = client.get("/v1/models", headers=AUTH_HEADERS)
            assert first.status_code == 200
            assert [m["id"] for m in first.json()["data"]] == ["m1", "m2"]
            assert second.json() == first.json()
        models_hits = [c for c in calls if c["url"].endswith("/v1/models")]
        assert len(models_hits) == 1, "second call should hit the cache"


class TestChatCompletions:
    def test_non_stream_plain_chat(self):
        calls: list[dict] = []
        with setup_test_client(calls) as client:
            resp = client.post(
                "/v1/chat/completions",
                headers=AUTH_HEADERS,
                json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["choices"][0]["message"]["content"] == "hi there"
        assert body["usage"] == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
        # upstream received v3 body with prompt + toolChoice shape
        upstream = calls[0]["body"]
        assert upstream["prompt"][0]["content"] == [{"type": "text", "text": "hi"}]
        assert upstream["toolChoice"] == {"type": "auto"}

    def test_stream_returns_sse(self):
        calls: list[dict] = []
        with setup_test_client(calls) as client:
            resp = client.post(
                "/v1/chat/completions",
                headers=AUTH_HEADERS,
                json={"model": "gpt-4", "stream": True,
                      "messages": [{"role": "user", "content": "hi"}]},
            )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        text = resp.text
        assert 'data: {"type":"text-delta' not in text  # translated, not raw v3
        assert '"content": "hi"' in text
        assert '"content": " there"' in text
        assert "data: [DONE]" in text
        # usage present by default on the finish chunk
        finish = [l for l in text.split("\n\n")
                  if l.startswith("data:") and "finish_reason" in l]
        assert finish and '"usage"' in finish[-1]

    def test_stream_include_usage_false(self):
        with setup_test_client([]) as client:
            resp = client.post(
                "/v1/chat/completions",
                headers=AUTH_HEADERS,
                json={"model": "gpt-4", "stream": True,
                      "stream_options": {"include_usage": False},
                      "messages": [{"role": "user", "content": "hi"}]},
            )
        finish = [l for l in resp.text.split("\n\n")
                  if l.startswith("data:") and "finish_reason" in l]
        assert finish and '"usage"' not in finish[-1]

    def test_tool_history_validation_400(self):
        with setup_test_client([]) as client:
            resp = client.post(
                "/v1/chat/completions",
                headers=AUTH_HEADERS,
                json={"model": "gpt-4", "messages": [
                    {"role": "user", "content": "go"},
                    {"role": "tool", "tool_call_id": "x", "content": "orphan"},
                ]},
            )
        assert resp.status_code == 400
        err = resp.json()["error"]
        assert err["type"] == "invalid_request_error"
        assert "no preceding assistant" in err["message"]

    def test_tool_call_round_trip_non_stream(self):
        calls: list[dict] = []
        with setup_test_client(calls) as client:
            resp = client.post(
                "/v1/chat/completions",
                headers=AUTH_HEADERS,
                json={"model": "gpt-4", "messages": [
                    {"role": "user", "content": "run"},
                    {"role": "assistant", "content": None, "tool_calls": [{
                        "id": "call_1", "type": "function",
                        "function": {"name": "calc", "arguments": '{"a":1}'},
                    }]},
                    {"role": "tool", "tool_call_id": "call_1", "content": "result"},
                ], "tools": [{
                    "type": "function",
                    "function": {"name": "calc", "parameters": {"type": "object"}},
                }]},
            )
        assert resp.status_code == 200
        msg = resp.json()["choices"][0]["message"]
        assert msg["tool_calls"][0]["function"]["name"] == "calc"
        assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"a": 1}
        # upstream got fx wire format: assistant tool-call as a content part
        upstream = calls[0]["body"]
        assistant = upstream["prompt"][1]
        assert "toolCalls" not in assistant
        assert assistant["content"][1] == {
            "type": "tool-call", "toolCallId": "call_1", "toolName": "calc", "input": {"a": 1},
        }
        # backfilled toolName on the tool result
        assert upstream["prompt"][2]["content"][0]["toolName"] == "calc"

    def test_upstream_json_error_normalized(self):
        with setup_test_client([]) as client:
            resp = client.post(
                "/v1/chat/completions",
                headers=AUTH_HEADERS,
                json={"model": "gpt-4", "messages": [
                    {"role": "user", "content": "Invalid input please"},
                ]},
            )
        assert resp.status_code == 400
        body = resp.json()
        assert body["error"]["message"] == "Invalid input"
        assert body["error"]["path"] == ["toolChoice"]  # detail preserved

    def test_plain_text_error_shape(self):
        calls: list[dict] = []
        server.app.state.client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda req: httpx.Response(502, text="upstream exploded")),
            timeout=httpx.Timeout(5.0),
        )
        server.app.state.models_cache = {"data": None, "expires": 0.0}
        with TestClient(server.app) as client:
            resp = client.post(
                "/v1/chat/completions",
                headers=AUTH_HEADERS,
                json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
            )
        assert resp.status_code == 502
        assert resp.json()["error"]["message"] == "upstream exploded"

    def test_invalid_json_body_400(self):
        with setup_test_client([]) as client:
            resp = client.post(
                "/v1/chat/completions",
                headers=AUTH_HEADERS,
                content="not json",
            )
        assert resp.status_code == 400

    def test_missing_model_uses_default(self):
        calls: list[dict] = []
        with setup_test_client(calls) as client:
            resp = client.post(
                "/v1/chat/completions",
                headers=AUTH_HEADERS,
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
        assert resp.status_code == 200
        # model travels in the ai-language-model-id header, defaulted here
        model_id = calls[0]["headers"].get("ai-language-model-id")
        assert model_id == server.DEFAULT_MODEL


class TestResponses:
    def test_non_stream(self):
        calls: list[dict] = []
        with setup_test_client(calls) as client:
            resp = client.post(
                "/v1/responses",
                headers=AUTH_HEADERS,
                json={"model": "gpt-4", "input": [{"role": "user", "content": "hi"}]},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "response"
        assert body["status"] == "completed"
        assert body["output"][0]["type"] == "message"
        assert body["output"][0]["content"][0]["text"] == "hi there"

    def test_stream(self):
        with setup_test_client([]) as client:
            resp = client.post(
                "/v1/responses",
                headers=AUTH_HEADERS,
                json={"model": "gpt-4", "stream": True,
                      "input": [{"role": "user", "content": "hi"}]},
            )
        assert resp.status_code == 200
        text = resp.text
        assert "response.created" in text
        assert "response.output_text.delta" in text
        assert "response.completed" in text
        assert "data: [DONE]" in text

    def test_tool_input_validation_400(self):
        with setup_test_client([]) as client:
            resp = client.post(
                "/v1/responses",
                headers=AUTH_HEADERS,
                json={"model": "gpt-4", "input": [
                    {"type": "function_call_output", "call_id": "c1", "output": "x"},
                ]},
            )
        assert resp.status_code == 400
        assert "Invalid input" in resp.json()["error"]["message"]


class TestEmbeddings:
    def test_embeddings_forwarded(self):
        with setup_test_client([]) as client:
            resp = client.post(
                "/v1/embeddings",
                headers=AUTH_HEADERS,
                json={"model": "openai/text-embedding-3-large",
                      "input": ["hello"]},
            )
        assert resp.status_code == 200
        assert resp.json()["data"][0]["embedding"] == [0.1, 0.2]


# =====================================================================
# Anthropic Messages API (/v1/messages)
# =====================================================================


ANTHROPIC_AUTH_HEADERS = {"x-api-key": "test-proxy-key"}


class TestAnthropicAuth:
    def test_no_key_rejected(self):
        with setup_test_client([]) as client:
            resp = client.post("/v1/messages", json={
                "model": "claude-3", "max_tokens": 10,
                "messages": [{"role": "user", "content": "hi"}],
            })
        assert resp.status_code == 401
        assert resp.json()["error"]["type"] == "authentication_error"

    def test_wrong_key_rejected(self):
        with setup_test_client([]) as client:
            resp = client.post("/v1/messages", headers={"x-api-key": "nope"},
                               json={"model": "claude-3", "max_tokens": 10,
                                     "messages": [{"role": "user", "content": "hi"}]})
        assert resp.status_code == 401

    def test_bearer_token_also_accepted(self):
        with setup_test_client([]) as client:
            resp = client.post(
                "/v1/messages",
                headers=AUTH_HEADERS,  # Authorization: Bearer test-proxy-key
                json={"model": "claude-3", "max_tokens": 10,
                      "messages": [{"role": "user", "content": "hi"}]},
            )
        assert resp.status_code == 200


class TestAnthropicMessages:
    def test_non_stream_plain_text(self):
        calls: list[dict] = []
        with setup_test_client(calls) as client:
            resp = client.post(
                "/v1/messages",
                headers=ANTHROPIC_AUTH_HEADERS,
                json={
                    "model": "claude-3", "max_tokens": 100,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["type"] == "message"
        assert body["role"] == "assistant"
        assert body["content"] == [{"type": "text", "text": "hi there"}]
        assert body["stop_reason"] == "end_turn"
        assert body["usage"]["input_tokens"] == 3
        assert body["usage"]["output_tokens"] == 2
        # upstream received v3 body with translated prompt
        upstream = calls[0]["body"]
        assert upstream["prompt"][0]["content"] == [{"type": "text", "text": "hi"}]
        assert upstream["toolChoice"] == {"type": "auto"}

    def test_system_prompt_translated(self):
        calls: list[dict] = []
        with setup_test_client(calls) as client:
            resp = client.post(
                "/v1/messages",
                headers=ANTHROPIC_AUTH_HEADERS,
                json={
                    "model": "claude-3", "max_tokens": 100,
                    "system": "be nice",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        assert resp.status_code == 200
        upstream = calls[0]["body"]
        # system should be the first prompt entry as a plain string
        assert upstream["prompt"][0] == {"role": "system", "content": "be nice"}
        assert upstream["prompt"][1]["content"] == [{"type": "text", "text": "hi"}]

    def test_stream_returns_anthropic_sse(self):
        with setup_test_client([]) as client:
            resp = client.post(
                "/v1/messages",
                headers=ANTHROPIC_AUTH_HEADERS,
                json={
                    "model": "claude-3", "max_tokens": 100, "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        text = resp.text
        assert "event: message_start" in text
        assert "event: content_block_start" in text
        assert "event: content_block_delta" in text
        assert "text_delta" in text
        assert "event: content_block_stop" in text
        assert "event: message_delta" in text
        assert "event: message_stop" in text
        # actual content present
        assert '"text": "hi"' in text
        assert '"text": " there"' in text

    def test_tool_use_round_trip_non_stream(self):
        calls: list[dict] = []
        with setup_test_client(calls) as client:
            resp = client.post(
                "/v1/messages",
                headers=ANTHROPIC_AUTH_HEADERS,
                json={
                    "model": "claude-3", "max_tokens": 100,
                    "messages": [
                        {"role": "user", "content": "run"},
                        {"role": "assistant", "content": [
                            {"type": "tool_use", "id": "toolu_1",
                             "name": "calc", "input": {"a": 1}},
                        ]},
                        {"role": "user", "content": [
                            {"type": "tool_result", "tool_use_id": "toolu_1",
                             "content": "result"},
                        ]},
                    ],
                    "tools": [{
                        "name": "calc",
                        "description": "calculator",
                        "input_schema": {"type": "object"},
                    }],
                },
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["content"][0]["type"] == "tool_use"
        assert body["content"][0]["name"] == "calc"
        assert body["content"][0]["input"] == {"a": 1}
        assert body["stop_reason"] == "tool_use"
        # upstream got the tool as a content part in fx wire format
        upstream = calls[0]["body"]
        assert upstream["tools"][0]["name"] == "calc"
        assert upstream["tools"][0]["inputSchema"] == {"type": "object"}

    def test_missing_max_tokens_400(self):
        with setup_test_client([]) as client:
            resp = client.post(
                "/v1/messages",
                headers=ANTHROPIC_AUTH_HEADERS,
                json={"model": "claude-3", "messages": [{"role": "user", "content": "hi"}]},
            )
        assert resp.status_code == 400
        assert resp.json()["error"]["type"] == "invalid_request_error"

    def test_upstream_error_normalized_to_anthropic_shape(self):
        with setup_test_client([]) as client:
            resp = client.post(
                "/v1/messages",
                headers=ANTHROPIC_AUTH_HEADERS,
                json={"model": "claude-3", "max_tokens": 10,
                      "messages": [{"role": "user", "content": "Invalid input please"}]},
            )
        assert resp.status_code == 400
        body = resp.json()
        assert body["type"] == "error"
        assert body["error"]["message"] == "Invalid input"

    def test_invalid_json_body_400(self):
        with setup_test_client([]) as client:
            resp = client.post(
                "/v1/messages",
                headers=ANTHROPIC_AUTH_HEADERS,
                content="not json",
            )
        assert resp.status_code == 400

    def test_missing_model_uses_default(self):
        calls: list[dict] = []
        with setup_test_client(calls) as client:
            resp = client.post(
                "/v1/messages",
                headers=ANTHROPIC_AUTH_HEADERS,
                json={"max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]},
            )
        assert resp.status_code == 200
        model_id = calls[0]["headers"].get("ai-language-model-id")
        assert model_id == server.DEFAULT_MODEL


class TestAnthropicCountTokens:
    def test_count_tokens(self):
        with setup_test_client([]) as client:
            resp = client.post(
                "/v1/messages/count_tokens",
                headers=ANTHROPIC_AUTH_HEADERS,
                json={
                    "model": "claude-3", "max_tokens": 10,
                    "messages": [{"role": "user", "content": "hello world"}],
                },
            )
        assert resp.status_code == 200
        body = resp.json()
        assert "input_tokens" in body
        assert isinstance(body["input_tokens"], int)
        assert body["input_tokens"] > 0

    def test_count_tokens_with_system(self):
        with setup_test_client([]) as client:
            resp = client.post(
                "/v1/messages/count_tokens",
                headers=ANTHROPIC_AUTH_HEADERS,
                json={
                    "model": "claude-3", "max_tokens": 10,
                    "system": "you are a bot",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        assert resp.status_code == 200
        assert resp.json()["input_tokens"] > 0


# =====================================================================
# Reasoning / thinking end-to-end tests
# =====================================================================


class TestReasoningChatCompletions:
    """Reasoning flows through /v1/chat/completions (OpenAI format)."""

    def test_reasoning_non_stream(self):
        calls: list[dict] = []
        with setup_test_client(calls) as client:
            resp = client.post(
                "/v1/chat/completions",
                headers=AUTH_HEADERS,
                json={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "think"}],
                    "reasoning": "high",
                },
            )
        assert resp.status_code == 200
        msg = resp.json()["choices"][0]["message"]
        assert msg["reasoning_content"] == "let me think harder"
        assert msg["content"] == "the answer"
        # v3 body got reasoning field
        upstream = calls[0]["body"]
        assert upstream["reasoning"] == "high"

    def test_reasoning_stream(self):
        with setup_test_client([]) as client:
            resp = client.post(
                "/v1/chat/completions",
                headers=AUTH_HEADERS,
                json={
                    "model": "gpt-4", "stream": True,
                    "messages": [{"role": "user", "content": "think"}],
                    "reasoning": "high",
                },
            )
        assert resp.status_code == 200
        text = resp.text
        # reasoning_content in the OpenAI stream chunks
        assert "reasoning_content" in text
        assert "let me think" in text
        assert "the answer" in text in text

    def test_reasoning_effort_normalized(self):
        calls: list[dict] = []
        with setup_test_client(calls) as client:
            resp = client.post(
                "/v1/chat/completions",
                headers=AUTH_HEADERS,
                json={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "think"}],
                    "reasoning_effort": "max",
                },
            )
        assert resp.status_code == 200
        upstream = calls[0]["body"]
        assert upstream["reasoning"] == "max"


class TestAnthropicReasoning:
    """Reasoning flows through /v1/messages (Anthropic format)."""

    def test_thinking_enabled_routes_reasoning_upstream(self):
        calls: list[dict] = []
        with setup_test_client(calls) as client:
            resp = client.post(
                "/v1/messages",
                headers=ANTHROPIC_AUTH_HEADERS,
                json={
                    "model": "claude-3", "max_tokens": 100,
                    "thinking": {"type": "enabled", "budget_tokens": 10000},
                    "messages": [{"role": "user", "content": "think"}],
                },
            )
        assert resp.status_code == 200
        upstream = calls[0]["body"]
        assert upstream["reasoning"] == "enabled"

    def test_thinking_non_stream_returns_thinking_block(self):
        with setup_test_client([]) as client:
            resp = client.post(
                "/v1/messages",
                headers=ANTHROPIC_AUTH_HEADERS,
                json={
                    "model": "claude-3", "max_tokens": 100,
                    "thinking": {"type": "enabled", "budget_tokens": 10000},
                    "messages": [{"role": "user", "content": "think"}],
                },
            )
        assert resp.status_code == 200
        body = resp.json()
        types = [c["type"] for c in body["content"]]
        assert types == ["thinking", "text"]
        assert body["content"][0]["thinking"] == "let me think harder"
        assert body["content"][1]["text"] == "the answer"

    def test_thinking_stream_returns_thinking_events(self):
        with setup_test_client([]) as client:
            resp = client.post(
                "/v1/messages",
                headers=ANTHROPIC_AUTH_HEADERS,
                json={
                    "model": "claude-3", "max_tokens": 100, "stream": True,
                    "thinking": {"type": "enabled", "budget_tokens": 10000},
                    "messages": [{"role": "user", "content": "think"}],
                },
            )
        assert resp.status_code == 200
        text = resp.text
        assert "event: message_start" in text
        assert "thinking_delta" in text
        assert "text_delta" in text
        # thinking block should be closed before text block starts
        assert text.index("content_block_stop") < text.index("text_delta")
        assert "event: message_stop" in text

    def test_no_thinking_no_reasoning_upstream(self):
        calls: list[dict] = []
        with setup_test_client(calls) as client:
            resp = client.post(
                "/v1/messages",
                headers=ANTHROPIC_AUTH_HEADERS,
                json={
                    "model": "claude-3", "max_tokens": 100,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        assert resp.status_code == 200
        upstream = calls[0]["body"]
        assert "reasoning" not in upstream


class TestTopK:
    def test_top_k_in_v3_body(self):
        calls: list[dict] = []
        with setup_test_client(calls) as client:
            resp = client.post(
                "/v1/chat/completions",
                headers=AUTH_HEADERS,
                json={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "hi"}],
                    "top_k": 40,
                },
            )
        assert resp.status_code == 200
        upstream = calls[0]["body"]
        assert upstream["topK"] == 40

    def test_top_k_via_anthropic(self):
        calls: list[dict] = []
        with setup_test_client(calls) as client:
            resp = client.post(
                "/v1/messages",
                headers=ANTHROPIC_AUTH_HEADERS,
                json={
                    "model": "claude-3", "max_tokens": 100,
                    "top_k": 40,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        assert resp.status_code == 200
        upstream = calls[0]["body"]
        assert upstream["topK"] == 40
