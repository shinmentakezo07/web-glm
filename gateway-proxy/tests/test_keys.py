"""Tests for the multi-key pool: loading, sticky-key selection, failover, cooldown."""

import json
import os

import httpx
import pytest
from starlette.testclient import TestClient

os.environ["PROXY_API_KEY"] = "test-proxy-key"
os.environ["GATEWAY_HTTP2"] = "0"

import server  # noqa: E402
from keys import KeyPool, load_keys  # noqa: E402


# --------------------------------------------------------------------------- #
# load_keys
# --------------------------------------------------------------------------- #


class TestLoadKeys:
    def test_numbered_vars(self, monkeypatch):
        for name in list(os.environ):
            if name.startswith("AI_GATEWAY_API_KEY"):
                monkeypatch.delenv(name)
        monkeypatch.setenv("AI_GATEWAY_API_KEY_1", "k1")
        monkeypatch.setenv("AI_GATEWAY_API_KEY_2", "k2")
        assert load_keys() == ["k1", "k2"]

    def test_legacy_var_is_first(self, monkeypatch):
        for name in list(os.environ):
            if name.startswith("AI_GATEWAY_API_KEY"):
                monkeypatch.delenv(name)
        monkeypatch.setenv("AI_GATEWAY_API_KEY", "legacy")
        monkeypatch.setenv("AI_GATEWAY_API_KEY_1", "k1")
        assert load_keys() == ["legacy", "k1"]

    def test_comma_separated_legacy(self, monkeypatch):
        for name in list(os.environ):
            if name.startswith("AI_GATEWAY_API_KEY"):
                monkeypatch.delenv(name)
        monkeypatch.setenv("AI_GATEWAY_API_KEY", "a, b ,,c")
        assert load_keys() == ["a", "b", "c"]

    def test_duplicates_dropped(self, monkeypatch):
        for name in list(os.environ):
            if name.startswith("AI_GATEWAY_API_KEY"):
                monkeypatch.delenv(name)
        monkeypatch.setenv("AI_GATEWAY_API_KEY", "dup")
        monkeypatch.setenv("AI_GATEWAY_API_KEY_1", "dup")
        monkeypatch.setenv("AI_GATEWAY_API_KEY_2", "other")
        assert load_keys() == ["dup", "other"]

    def test_empty(self, monkeypatch):
        for name in list(os.environ):
            if name.startswith("AI_GATEWAY_API_KEY"):
                monkeypatch.delenv(name)
        assert load_keys() == []


# --------------------------------------------------------------------------- #
# KeyPool behaviour
# --------------------------------------------------------------------------- #


class TestStickySelection:
    def test_always_first_key_until_it_fails(self):
        # The first key is preferred and reused for every request; there is
        # no round-robin distribution across healthy keys.
        pool = KeyPool(["k1", "k2", "k3"], failover=True)
        got = [pool.next() for _ in range(6)]
        assert got == ["k1"] * 6

    def test_single_key_stable(self):
        pool = KeyPool(["only"], failover=True)
        assert all(pool.next() == "only" for _ in range(3))

    def test_empty_pool_returns_none(self):
        assert KeyPool([]).next() is None

    def test_exclude_skips_tried_keys(self):
        pool = KeyPool(["k1", "k2"], failover=False)
        assert pool.next(exclude={"k1"}) == "k2"
        assert pool.next(exclude={"k1", "k2"}) is None


class TestFailoverPolicy:
    def test_key_attributable_statuses(self):
        pool = KeyPool(["k1"], failover=True)
        for status in (401, 402, 403, 408, 429, 500, 502, 503, 504):
            assert pool.should_failover(status), status

    def test_request_faults_do_not_failover(self):
        pool = KeyPool(["k1"], failover=True)
        for status in (400, 404, 413, 422):
            assert not pool.should_failover(status), status

    def test_failover_disabled(self):
        pool = KeyPool(["k1"], failover=False)
        assert not pool.should_failover(429)


class TestCooldown:
    def test_failed_key_sits_out_then_recovers(self):
        pool = KeyPool(["k1", "k2"], failover=True, cooldown_seconds=60.0)
        assert pool.next() == "k1"
        pool.report_failure("k1")
        # k1 cooling -> next picks k2, then k2 again (k1 still cooling)...
        assert pool.next() == "k2"
        assert pool.next() == "k2"
        # ...unless everything else is excluded
        assert pool.next(exclude={"k2"}) == "k1"
        pool.report_success("k1")
        assert pool.next(exclude={"k2"}) == "k1"

    def test_cooldown_zero_disables(self):
        pool = KeyPool(["k1", "k2"], failover=True, cooldown_seconds=0.0)
        assert pool.next() == "k1"
        pool.report_failure("k1")
        # cooldown disabled -> k1 is not skipped, stays the preferred key
        assert pool.next() == "k1"


class TestCoolingKeys:
    def test_fresh_pool_has_none(self):
        pool = KeyPool(["k1", "k2"], cooldown_seconds=60.0)
        assert pool.cooling_keys() == []

    def test_failed_key_is_listed(self):
        pool = KeyPool(["k1", "k2"], cooldown_seconds=60.0)
        pool.report_failure("k2")
        assert pool.cooling_keys() == ["k2"]

    def test_success_clears_entry(self):
        pool = KeyPool(["k1"], cooldown_seconds=60.0)
        pool.report_failure("k1")
        pool.report_success("k1")
        assert pool.cooling_keys() == []

    def test_zero_cooldown_never_lists(self):
        pool = KeyPool(["k1"], cooldown_seconds=0.0)
        pool.report_failure("k1")
        assert pool.cooling_keys() == []

    def test_lists_follow_pool_order(self):
        pool = KeyPool(["a", "b", "c"], cooldown_seconds=60.0)
        pool.report_failure("c")
        pool.report_failure("a")
        assert pool.cooling_keys() == ["a", "c"]


# --------------------------------------------------------------------------- #
# Integration through the FastAPI app
# --------------------------------------------------------------------------- #

AUTH_HEADERS = {"Authorization": "Bearer test-proxy-key"}

KEY1, KEY2 = "test-gateway-key-1", "test-gateway-key-2"


def sse_chat(body: dict) -> httpx.Response:
    payload = "\n".join(
        [
            'data: {"type":"start"}',
            'data: {"type":"text-delta","delta":"hi"}',
            "data: [DONE]",
        ]
    )
    return httpx.Response(200, text=payload, headers={"content-type": "text/event-stream"})


class RecordingRouter(httpx.MockTransport):
    """Serves chat responses; fails with `fail_status` while a key is on turn N."""

    def __init__(self, calls: list[dict], fail_status: int | None, fail_auth: str | None):
        self.calls = calls
        self.fail_status = fail_status
        self.fail_auth = fail_auth
        super().__init__(self.handler)

    def handler(self, request: httpx.Request) -> httpx.Response:
        auth = request.headers.get("authorization", "")
        calls = self.calls
        calls.append({"auth": auth, "path": request.url.path})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "m1"}]})
        if request.url.path == "/v3/ai/language-model":
            if self.fail_status and auth == f"Bearer {self.fail_auth}":
                return httpx.Response(self.fail_status, json={"error": {"message": "nope"}})
            return sse_chat({})
        return httpx.Response(404, json={"error": {"message": "nf"}})


def setup(calls: list[dict], pool: KeyPool, router: RecordingRouter) -> TestClient:
    server.KEY_POOL = pool
    server.app.state.client = httpx.AsyncClient(transport=router, timeout=httpx.Timeout(5.0))
    server.app.state.models_cache = {"data": None, "expires": 0.0}
    return TestClient(server.app)


CHAT_BODY = {
    "model": "zai/glm-5.2",
    "messages": [{"role": "user", "content": "Say hi"}],
}


class TestChatFailover:
    @pytest.mark.parametrize("status", [401, 402, 403, 429, 500])
    def test_fails_over_to_second_key(self, status):
        calls: list[dict] = []
        router = RecordingRouter(calls, fail_status=status, fail_auth=KEY1)
        pool = KeyPool([KEY1, KEY2], failover=True, cooldown_seconds=0.0)
        with setup(calls, pool, router) as client:
            resp = client.post("/v1/chat/completions", headers=AUTH_HEADERS, json=CHAT_BODY)
        assert resp.status_code == 200
        assert [c["auth"] for c in calls] == [
            f"Bearer {KEY1}",
            f"Bearer {KEY2}",
        ], "first attempt must use key1, retry key2"

    def test_no_failover_on_bad_request(self):
        calls: list[dict] = []
        router = RecordingRouter(calls, fail_status=400, fail_auth=KEY1)
        pool = KeyPool([KEY1, KEY2], failover=True, cooldown_seconds=0.0)
        with setup(calls, pool, router) as client:
            resp = client.post("/v1/chat/completions", headers=AUTH_HEADERS, json=CHAT_BODY)
        assert resp.status_code == 400
        assert len(calls) == 1, "request faults must not burn another key"

    def test_failover_disabled_by_flag(self):
        calls: list[dict] = []
        router = RecordingRouter(calls, fail_status=429, fail_auth=KEY1)
        pool = KeyPool([KEY1, KEY2], failover=False, cooldown_seconds=0.0)
        with setup(calls, pool, router) as client:
            resp = client.post("/v1/chat/completions", headers=AUTH_HEADERS, json=CHAT_BODY)
        assert resp.status_code == 429
        assert len(calls) == 1

    def test_sticky_key_reused_across_requests(self):
        # With no failures the first key is reused for every request;
        # there is no round-robin alternation between healthy keys.
        calls: list[dict] = []
        router = RecordingRouter(calls, fail_status=None, fail_auth=None)
        pool = KeyPool([KEY1, KEY2], failover=True, cooldown_seconds=0.0)
        with setup(calls, pool, router) as client:
            for _ in range(4):
                resp = client.post("/v1/chat/completions", headers=AUTH_HEADERS, json=CHAT_BODY)
                assert resp.status_code == 200
        assert [c["auth"] for c in calls] == [f"Bearer {KEY1}"] * 4

    def test_streaming_failover_before_first_chunk(self):
        calls: list[dict] = []
        router = RecordingRouter(calls, fail_status=402, fail_auth=KEY1)
        pool = KeyPool([KEY1, KEY2], failover=True, cooldown_seconds=0.0)
        body = dict(CHAT_BODY, stream=True)
        with setup(calls, pool, router) as client:
            resp = client.post("/v1/chat/completions", headers=AUTH_HEADERS, json=body)
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        assert [c["auth"] for c in calls] == [f"Bearer {KEY1}", f"Bearer {KEY2}"]

    def test_all_keys_dead_returns_last_error(self):
        calls: list[dict] = []
        router = RecordingRouter(calls, fail_status=429, fail_auth="ANY")
        # make every key fail regardless of which one is used
        router.fail_auth = None
        pool = KeyPool([KEY1, KEY2], failover=True, cooldown_seconds=0.0)

        def always_fail(request: httpx.Request) -> httpx.Response:
            calls.append({"path": request.url.path})
            return httpx.Response(429, json={"error": {"message": "rate limited"}})

        server.KEY_POOL = pool
        server.app.state.client = httpx.AsyncClient(
            transport=httpx.MockTransport(always_fail), timeout=httpx.Timeout(5.0)
        )
        server.app.state.models_cache = {"data": None, "expires": 0.0}
        try:
            client = TestClient(server.app)
            resp = client.post("/v1/chat/completions", headers=AUTH_HEADERS, json=CHAT_BODY)
        finally:
            pass
        assert resp.status_code == 429
        assert len(calls) == 2, "both keys tried exactly once"


class TestOtherRoutesFailover:
    def test_models_failover(self):
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            auth = request.headers.get("authorization", "")
            seen.append(auth)
            if auth.endswith(KEY1):
                return httpx.Response(401, json={"error": {"message": "bad key"}})
            return httpx.Response(200, json={"data": [{"id": "m1"}]})

        pool = KeyPool([KEY1, KEY2], failover=True, cooldown_seconds=0.0)
        server.KEY_POOL = pool
        server.app.state.client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler), timeout=httpx.Timeout(5.0)
        )
        server.app.state.models_cache = {"data": None, "expires": 0.0}
        with TestClient(server.app) as client:
            resp = client.get("/v1/models", headers=AUTH_HEADERS)
        assert resp.status_code == 200
        assert seen == [f"Bearer {KEY1}", f"Bearer {KEY2}"]

    def test_embeddings_failover(self):
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            auth = request.headers.get("authorization", "")
            seen.append(auth)
            if auth.endswith(KEY1):
                return httpx.Response(429, json={"error": {"message": "slow down"}})
            return httpx.Response(200, json={"data": [{"embedding": [0.1]}]})

        pool = KeyPool([KEY1, KEY2], failover=True, cooldown_seconds=0.0)
        server.KEY_POOL = pool
        server.app.state.client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler), timeout=httpx.Timeout(5.0)
        )
        server.app.state.models_cache = {"data": None, "expires": 0.0}
        with TestClient(server.app) as client:
            resp = client.post(
                "/v1/embeddings",
                headers=AUTH_HEADERS,
                json={"model": "openai/text-embedding-3-large", "input": "x"},
            )
        assert resp.status_code == 200
        assert seen == [f"Bearer {KEY1}", f"Bearer {KEY2}"]


class TestHealthReportsPool:
    def test_healthz_lists_key_count(self):
        pool = KeyPool([KEY1, KEY2], failover=True, cooldown_seconds=30.0)
        server.KEY_POOL = pool
        server.app.state.client = httpx.AsyncClient(timeout=httpx.Timeout(5.0))
        server.app.state.models_cache = {"data": None, "expires": 0.0}
        with TestClient(server.app) as client:
            resp = client.get("/healthz")
        assert resp.status_code == 200
        data = resp.json()
        assert data["keys"]["count"] == 2
        assert data["keys"]["failover"] is True
        assert "rotation" not in data["keys"]
        # never leak raw keys through health
        assert KEY1 not in json.dumps(data)
