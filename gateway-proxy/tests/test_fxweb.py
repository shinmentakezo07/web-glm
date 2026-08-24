"""Tests for the fx.sh free web-gateway fallback provider.

The fx.sh endpoint (https://fx.sh/fx-wasm/gateway/v3/ai/language-model) is a
free, no-API-key alternative to the Vercel AI Gateway. It speaks the same v3
AI SDK protocol. These tests verify:

  - The fx.sh fallback fires when no Vercel keys are configured.
  - The fx.sh fallback fires after all Vercel keys fail over.
  - The fx.sh headers match the HAR capture (no Authorization, browser UA,
    Origin/Referer, session pinning).
  - The fx.sh URL and path are correct.
  - FXWEB_FALLBACK=0 disables the fallback.
  - The v3 body sent to fx.sh is identical to the one sent to Vercel.
"""

import json
import os
import tempfile

os.environ["PROXY_API_KEY"] = "test-proxy-key"
os.environ["AI_GATEWAY_API_KEY"] = "test-gateway-key"
os.environ["GATEWAY_HTTP2"] = "0"
os.environ["MODELS_CACHE_TTL"] = "300"
os.environ["USAGE_TRACKING"] = "0"
os.environ["USAGE_DB_PATH"] = os.path.join(tempfile.gettempdir(), "test_usage.db")

import httpx
from fastapi.testclient import TestClient

import server


AUTH_HEADERS = {"Authorization": "Bearer test-proxy-key"}

V3_CHAT_BODY = {
    "model": "zai/glm-5.2",
    "messages": [{"role": "user", "content": "hi"}],
}


def _sse_response() -> httpx.Response:
    """A canned v3 SSE stream that the proxy can translate."""
    return httpx.Response(
        200,
        content=(
            'data: {"type":"text-delta","delta":"hello"}\n\n'
            'data: {"type":"finish","finishReason":"stop",'
            '"usage":{"inputTokens":{"total":2},"outputTokens":{"total":1}}}\n\n'
        ),
        headers={"content-type": "text/event-stream"},
    )


# --------------------------------------------------------------------------- #
# Header construction
# --------------------------------------------------------------------------- #


class TestFxwebHeaders:
    def test_authorization_header_present(self):
        """fx.sh uses the public demo key: Authorization: Bearer fx-demo-proxy."""
        headers = server._fxweb_headers("zai/glm-5.2")
        assert headers["Authorization"] == f"Bearer {server.FXWEB_API_KEY}"
        assert server.FXWEB_API_KEY == "fx-demo-proxy"

    def test_browser_user_agent(self):
        """HTTP-level User-Agent should be a browser, not fx/<version>."""
        headers = server._fxweb_headers("zai/glm-5.2")
        assert "Chrome" in headers["User-Agent"]
        assert not headers["User-Agent"].startswith("fx/")

    def test_protocol_headers_present(self):
        headers = server._fxweb_headers("zai/glm-5.2")
        assert headers["ai-gateway-protocol-version"] == server.identity.state["protocol_version"]
        assert headers["ai-language-model-specification-version"] == server.identity.state["specification_version"]
        assert headers["ai-language-model-id"] == "zai/glm-5.2"
        assert headers["ai-language-model-streaming"] == "true"

    def test_origin_and_referer(self):
        headers = server._fxweb_headers("zai/glm-5.2")
        assert headers["Origin"] == "https://fx.sh"
        assert headers["Referer"] == "https://fx.sh/"

    def test_sec_fetch_headers(self):
        """fx.sh edge requires sec-fetch-* same-origin headers (403 without)."""
        headers = server._fxweb_headers("zai/glm-5.2")
        assert headers["sec-fetch-dest"] == "empty"
        assert headers["sec-fetch-mode"] == "cors"
        assert headers["sec-fetch-site"] == "same-origin"

    def test_session_headers_auto_generated(self):
        """When no session_id is provided, one is generated in the fx.sh shape."""
        headers = server._fxweb_headers("zai/glm-5.2")
        sid = headers["x-session-id"]
        affinity = headers["x-session-affinity"]
        assert sid == affinity
        # fx.sh shape: <ms>-<ms*1000000>-<16hex>
        parts = sid.split("-")
        assert len(parts) == 3
        assert len(parts[2]) == 16  # 16 hex chars

    def test_session_headers_from_param(self):
        headers = server._fxweb_headers(
            "zai/glm-5.2", session_id="mysess", session_affinity="mysess",
        )
        assert headers["x-session-id"] == "mysess"
        assert headers["x-session-affinity"] == "mysess"

    def test_http_referer_and_x_title(self):
        headers = server._fxweb_headers("zai/glm-5.2")
        assert headers["HTTP-Referer"] == "https://github.com/vercel-labs/fx"
        assert headers["X-Title"] == "fx"


# --------------------------------------------------------------------------- #
# Session ID generation
# --------------------------------------------------------------------------- #


class TestFxwebSession:
    def test_generated_session_shape(self):
        sid, affinity = server._generate_fxweb_session()
        assert sid == affinity
        parts = sid.split("-")
        assert len(parts) == 3
        # part 0: millisecond timestamp
        int(parts[0])
        # part 1: ms * 1000000
        int(parts[1])
        # part 2: 16 hex chars
        assert len(parts[2]) == 16
        int(parts[2], 16)  # valid hex

    def test_generated_session_unique(self):
        sessions = {server._generate_fxweb_session()[0] for _ in range(20)}
        assert len(sessions) == 20, "each generated session must be unique"


# --------------------------------------------------------------------------- #
# End-to-end: fx.sh fallback via mocked transport
# --------------------------------------------------------------------------- #


def _make_router(calls: list[dict], fxweb_succeeds: bool = True):
    """Transport that routes Vercel paths and fx.sh paths differently."""

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append({
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "headers": dict(request.headers),
            "body": json.loads(request.content) if request.content else None,
        })
        path = request.url.path
        # fx.sh paths
        if path.startswith("/fx-wasm/gateway/"):
            if not fxweb_succeeds:
                return httpx.Response(429, json={"error": {"message": "fx.sh rate limited"}})
            if path.endswith("/v1/models"):
                return httpx.Response(200, json={"data": [{"id": "zai/glm-5.2"}]})
            return _sse_response()
        # Vercel paths
        if path == "/v3/ai/language-model":
            return httpx.Response(
                429, json={"error": {"message": "vercel rate limited"}}
            )
        if path == "/v1/models":
            return httpx.Response(
                429, json={"error": {"message": "vercel rate limited"}}
            )
        return httpx.Response(404, json={"error": {"message": "not found"}})

    return httpx.MockTransport(handler)


def _setup(calls: list[dict], *, fxweb_succeeds=True, pool_keys=None):
    server.USAGE.enabled = False
    # Reset fx.sh cooldown so tests don't leak state between each other.
    server._fxweb_cooldown_until = 0.0
    if pool_keys is not None:
        server.KEY_POOL = server.KeyPool(pool_keys)
    server.app.state.client = httpx.AsyncClient(
        transport=_make_router(calls, fxweb_succeeds=fxweb_succeeds),
        timeout=httpx.Timeout(5.0),
    )
    server.app.state.models_cache = {"data": None, "expires": 0.0}
    return TestClient(server.app)


class TestFxwebFallbackChat:
    def test_fallback_when_no_keys(self):
        """With no Vercel keys, the fx.sh fallback serves the request directly.

        When no keys are configured and fx.sh is available, the proxy skips
        the pointless unauthenticated Vercel attempt and goes straight to
        fx.sh.
        """
        calls: list[dict] = []
        server.KEY_POOL = server.KeyPool([])
        with _setup(calls) as client:
            resp = client.post(
                "/v1/chat/completions", headers=AUTH_HEADERS, json=V3_CHAT_BODY,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["choices"][0]["message"]["content"] == "hello"
        # Goes straight to fx.sh — no Vercel attempt.
        paths = [c["path"] for c in calls]
        assert "/fx-wasm/gateway/v3/ai/language-model" in paths
        assert "/v3/ai/language-model" not in paths

    def test_fallback_after_all_keys_fail(self):
        """When all Vercel keys fail, fx.sh fallback kicks in."""
        calls: list[dict] = []
        server.KEY_POOL = server.KeyPool(
            ["key-1", "key-2"], failover=True, cooldown_seconds=0.0,
        )
        with _setup(calls) as client:
            resp = client.post(
                "/v1/chat/completions", headers=AUTH_HEADERS, json=V3_CHAT_BODY,
            )
        assert resp.status_code == 200
        assert resp.json()["choices"][0]["message"]["content"] == "hello"
        paths = [c["path"] for c in calls]
        # Vercel tried first (both keys), then fx.sh
        assert paths == [
            "/v3/ai/language-model",
            "/v3/ai/language-model",
            "/fx-wasm/gateway/v3/ai/language-model",
        ]

    def test_fallback_headers_correct_auth(self):
        """The fx.sh request must carry Authorization: Bearer fx-demo-proxy."""
        calls: list[dict] = []
        server.KEY_POOL = server.KeyPool([])
        with _setup(calls) as client:
            client.post(
                "/v1/chat/completions", headers=AUTH_HEADERS, json=V3_CHAT_BODY,
            )
        fxweb_call = next(
            c for c in calls if "fx-wasm" in c["path"]
        )
        hdrs = {k.lower(): v for k, v in fxweb_call["headers"].items()}
        assert hdrs["authorization"] == "Bearer fx-demo-proxy"
        assert "Chrome" in hdrs["user-agent"]
        assert hdrs["origin"] == "https://fx.sh"
        assert hdrs["ai-language-model-streaming"] == "true"

    def test_fallback_sends_same_v3_body(self):
        """The v3 body sent to fx.sh must match the one sent to Vercel."""
        calls: list[dict] = []
        server.KEY_POOL = server.KeyPool(
            ["key-1"], failover=True, cooldown_seconds=0.0,
        )
        with _setup(calls) as client:
            client.post(
                "/v1/chat/completions", headers=AUTH_HEADERS, json=V3_CHAT_BODY,
            )
        vercel_body = calls[0]["body"]
        fxweb_body = calls[1]["body"]
        assert vercel_body == fxweb_body
        assert vercel_body["prompt"][0]["content"] == [{"type": "text", "text": "hi"}]
        assert vercel_body["toolChoice"] == {"type": "auto"}
        # body-level headers.user-agent present (fx product UA)
        assert "headers" in vercel_body
        assert vercel_body["headers"]["user-agent"].startswith("fx/")

    def test_fallback_streaming(self):
        """Streaming requests work through the fx.sh fallback."""
        calls: list[dict] = []
        server.KEY_POOL = server.KeyPool([])
        with _setup(calls) as client:
            resp = client.post(
                "/v1/chat/completions", headers=AUTH_HEADERS,
                json={**V3_CHAT_BODY, "stream": True},
            )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        assert '"content": "hello"' in resp.text

    def test_fallback_disabled(self):
        """When FXWEB_FALLBACK=0, the fallback is not tried."""
        calls: list[dict] = []
        server.KEY_POOL = server.KeyPool([])
        original = server.FXWEB_FALLBACK
        server.FXWEB_FALLBACK = False
        try:
            with _setup(calls) as client:
                resp = client.post(
                    "/v1/chat/completions", headers=AUTH_HEADERS, json=V3_CHAT_BODY,
                )
        finally:
            server.FXWEB_FALLBACK = original
        # No keys + fallback disabled → upstream error
        assert resp.status_code in (401, 429, 502, 500)
        # fx.sh was NOT called
        paths = [c["path"] for c in calls]
        assert "/fx-wasm/gateway/v3/ai/language-model" not in paths

    def test_fallback_also_fails_returns_error(self):
        """When both Vercel and fx.sh fail, the error is returned."""
        calls: list[dict] = []
        server.KEY_POOL = server.KeyPool(
            ["key-1"], failover=True, cooldown_seconds=0.0,
        )
        with _setup(calls, fxweb_succeeds=False) as client:
            resp = client.post(
                "/v1/chat/completions", headers=AUTH_HEADERS, json=V3_CHAT_BODY,
            )
        assert resp.status_code == 429
        paths = [c["path"] for c in calls]
        assert "/v3/ai/language-model" in paths
        assert "/fx-wasm/gateway/v3/ai/language-model" in paths

    def test_fxweb_wire_format_matches_har(self):
        """The fx.sh request must match the HAR capture exactly.

        Verifies every non-browser header from the HAR is present with the
        correct value, and that the v3 body has the body-level
        headers.user-agent = fx/<version> (just like the API-key path).
        """
        calls: list[dict] = []
        server.KEY_POOL = server.KeyPool([])
        with _setup(calls) as client:
            client.post(
                "/v1/chat/completions", headers=AUTH_HEADERS, json=V3_CHAT_BODY,
            )
        fxweb_call = next(c for c in calls if "fx-wasm" in c["path"])
        hdrs = {k.lower(): v for k, v in fxweb_call["headers"].items()}

        # --- Authorization: Bearer fx-demo-proxy (public demo key) ---
        # The HAR didn't show this because the browser's fetch adapter
        # injects it at the JS level (the WASM sets
        # AI_GATEWAY_API_KEY="fx-demo-proxy"). Our proxy sends it directly.
        assert hdrs["authorization"] == "Bearer fx-demo-proxy"

        # --- Protocol headers (same as Vercel + HAR) ---
        assert hdrs["ai-gateway-protocol-version"] == "0.0.1"
        assert hdrs["ai-language-model-specification-version"] == "4"
        assert hdrs["ai-language-model-id"] == "zai/glm-5.2"
        assert hdrs["ai-language-model-streaming"] == "true"

        # --- fx product identity headers ---
        assert hdrs["http-referer"] == "https://github.com/vercel-labs/fx"
        assert hdrs["x-title"] == "fx"

        # --- Browser-spoof headers (fx.sh is a web endpoint) ---
        assert "Chrome" in hdrs["user-agent"]
        assert hdrs["origin"] == "https://fx.sh"
        assert hdrs["referer"] == "https://fx.sh/"

        # --- Session pinning (auto-generated in fx.sh shape) ---
        sid = hdrs["x-session-id"]
        affinity = hdrs["x-session-affinity"]
        assert sid == affinity
        parts = sid.split("-")
        assert len(parts) == 3, "session id must be <ms>-<ms*1000000>-<hex16>"
        assert len(parts[2]) == 16, "nonce must be 16 hex chars"

        # --- v3 body: body-level headers.user-agent = fx/<version> ---
        # This is the SAME as the API-key path — the fx product version
        # lives in the body, not the HTTP header (just like the HAR).
        body = fxweb_call["body"]
        assert "headers" in body, "v3 body must include body-level headers"
        ua = body["headers"].get("user-agent", "")
        assert ua.startswith("fx/"), (
            f"body headers.user-agent must be fx/<version>, got: {ua!r}"
        )

        # --- v3 body: same shape as HAR ---
        assert body["toolChoice"] == {"type": "auto"}
        assert "tools" in body
        # user message content must be an array of parts (not a plain string)
        user_msg = next(m for m in body["prompt"] if m["role"] == "user")
        assert isinstance(user_msg["content"], list)
        assert user_msg["content"][0] == {"type": "text", "text": "hi"}


class TestFxwebFallbackModels:
    def test_models_fallback_when_no_keys(self):
        calls: list[dict] = []
        server.KEY_POOL = server.KeyPool([])
        with _setup(calls) as client:
            resp = client.get("/v1/models", headers=AUTH_HEADERS)
        assert resp.status_code == 200
        # fx.sh models endpoint was hit
        paths = [c["path"] for c in calls]
        assert "/fx-wasm/gateway/v1/models" in paths


class TestFxwebFallbackAnthropic:
    def test_anthropic_route_fallback(self):
        """/v1/messages also falls back to fx.sh."""
        calls: list[dict] = []
        server.KEY_POOL = server.KeyPool([])
        with _setup(calls) as client:
            resp = client.post(
                "/v1/messages",
                headers={"x-api-key": "test-proxy-key"},
                json={
                    "model": "zai/glm-5.2",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        assert resp.status_code == 200
        paths = [c["path"] for c in calls]
        assert "/fx-wasm/gateway/v3/ai/language-model" in paths


class TestFxwebHealthz:
    def test_healthz_reports_fallback(self):
        resp = None
        server.app.state.client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})),
            timeout=httpx.Timeout(5.0),
        )
        with TestClient(server.app) as client:
            resp = client.get("/healthz")
        assert resp.status_code == 200
        body = resp.json()
        assert "fxweb_fallback" in body
        assert body["fxweb_fallback"]["enabled"] is True
        assert body["fxweb_fallback"]["base_url"] == "https://fx.sh"
        assert "cooling" in body["fxweb_fallback"]


class TestFxwebCooldown:
    """When fx.sh returns 429 (demo rate limit), the fallback is put on
    cooldown so subsequent requests skip fx.sh entirely instead of
    wasting time hitting a rate-limited endpoint."""

    def setup_method(self):
        server._fxweb_cooldown_until = 0.0

    def teardown_method(self):
        server._fxweb_cooldown_until = 0.0

    def test_429_triggers_cooldown(self):
        """A 429 from fx.sh sets the cooldown timestamp."""
        server.KEY_POOL = server.KeyPool([])
        calls: list[dict] = []
        with _setup(calls, fxweb_succeeds=False) as client:
            resp = client.post(
                "/v1/chat/completions", headers=AUTH_HEADERS, json=V3_CHAT_BODY,
            )
        assert resp.status_code == 429
        assert server._fxweb_is_cooling()

    def test_cooldown_skips_fxweb(self):
        """When on cooldown, the fx.sh fallback is not tried."""
        server.KEY_POOL = server.KeyPool([])
        calls: list[dict] = []
        with _setup(calls) as client:
            server._fxweb_cooldown_until = float("inf")  # permanent cooldown
            resp = client.post(
                "/v1/chat/completions", headers=AUTH_HEADERS, json=V3_CHAT_BODY,
            )
        server._fxweb_cooldown_until = 0.0
        # fx.sh was NOT called (cooldown active)
        paths = [c["path"] for c in calls]
        assert "/fx-wasm/gateway/v3/ai/language-model" not in paths

    def test_second_request_after_429_skips_fxweb(self):
        """After a 429, the next request should NOT hit fx.sh."""
        server.KEY_POOL = server.KeyPool([])
        calls: list[dict] = []
        with _setup(calls, fxweb_succeeds=False) as client:
            # First request: hits fx.sh, gets 429, sets cooldown
            client.post("/v1/chat/completions", headers=AUTH_HEADERS, json=V3_CHAT_BODY)
            first_fxweb_hits = sum(1 for c in calls if "fx-wasm" in c["path"])
            # Second request: should skip fx.sh (cooldown active)
            client.post("/v1/chat/completions", headers=AUTH_HEADERS, json=V3_CHAT_BODY)
            second_fxweb_hits = sum(1 for c in calls if "fx-wasm" in c["path"])
        assert first_fxweb_hits == 1, "first request should hit fx.sh once"
        assert second_fxweb_hits == 1, "second request should NOT add fx.sh hits"

    def test_cooldown_can_be_disabled(self):
        """FXWEB_COOLDOWN=0 means no cooldown (always try fx.sh)."""
        original = server.FXWEB_COOLDOWN
        server.FXWEB_COOLDOWN = 0
        server.KEY_POOL = server.KeyPool([])
        calls: list[dict] = []
        try:
            with _setup(calls, fxweb_succeeds=False) as client:
                client.post("/v1/chat/completions", headers=AUTH_HEADERS, json=V3_CHAT_BODY)
                assert not server._fxweb_is_cooling()
                client.post("/v1/chat/completions", headers=AUTH_HEADERS, json=V3_CHAT_BODY)
        finally:
            server.FXWEB_COOLDOWN = original
            server._fxweb_cooldown_until = 0.0
        # Both requests hit fx.sh (no cooldown)
        fxweb_hits = sum(1 for c in calls if "fx-wasm" in c["path"])
        assert fxweb_hits == 2
