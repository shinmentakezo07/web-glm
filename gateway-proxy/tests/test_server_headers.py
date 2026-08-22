"""Unit tests for server.py header construction (env-free assertions)."""
from server import _v3_headers


def test_v3_headers_default_identity():
    headers = _v3_headers("zai/glm-5.2", streaming=True, api_key="test-gateway-key")
    assert headers["User-Agent"].startswith("fx/")
    assert headers["ai-language-model-streaming"] == "true"
    assert headers["ai-language-model-id"] == "zai/glm-5.2"
    assert headers["ai-gateway-protocol-version"] == "0.0.1"
    assert headers["ai-language-model-specification-version"] == "4"
    assert "x-session-id" not in headers
    assert "x-session-affinity" not in headers


def test_v3_headers_non_streaming_flag():
    headers = _v3_headers("zai/glm-5.2", streaming=False, api_key="test-gateway-key")
    assert headers["ai-language-model-streaming"] == "false"
    assert "Accept" not in headers


def test_v3_headers_session_params():
    headers = _v3_headers(
        "zai/glm-5.2", streaming=True, api_key="test-gateway-key",
        session_id="sess-1", session_affinity="sess-1",
    )
    assert headers["x-session-id"] == "sess-1"
    assert headers["x-session-affinity"] == "sess-1"
