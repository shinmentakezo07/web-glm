"""Tests for fx identity auto-sync (identity module).

The proxy must mirror the released fx binary's wire identity: the
``fx/<version>`` User-Agent and the ``ai-gateway-protocol-version`` /
``ai-language-model-specification-version`` headers. identity.py fetches them
from the vercel-labs/fx repo (releases API + raw Zig source) and hot-swaps
module state; server.py reads that state on every upstream request.
"""
import asyncio

import httpx
import pytest

import identity
from identity import apply, parse_release_version, parse_source_versions, refresh


@pytest.fixture(autouse=True)
def _restore_state():
    saved = dict(identity.state)
    yield
    identity.state.clear()
    identity.state.update(saved)


# --------------------------------------------------------------------------- #
# Parsing: raw fx Zig source
# --------------------------------------------------------------------------- #

RAW_SOURCE = """
const extra_headers = [_]std.http.Client.Request.ExtraHeaders{
    .{ .name = "Accept", .value = "application/json" },
    .{ .name = "ai-gateway-protocol-version", .value = "0.0.2" },
    .{ .name = "ai-language-model-specification-version", .value = "5" },
    .{ .name = "ai-language-model-id", .value = model },
};
"""


def test_parse_source_versions_extracts_header_values():
    assert parse_source_versions(RAW_SOURCE) == ("0.0.2", "5")


def test_parse_source_versions_tolerates_multiline_layout():
    text = 'buf[len] = .{ .name = "ai-gateway-protocol-version",\n .value = "9" };\n' \
           '.{ .name = "ai-language-model-specification-version", .value = "7" };'
    assert parse_source_versions(text) == ("9", "7")


def test_parse_source_versions_returns_none_on_unrelated_text():
    assert parse_source_versions("<html>404 Not Found</html>") is None


# --------------------------------------------------------------------------- #
# Parsing: releases API payload
# --------------------------------------------------------------------------- #


def test_parse_release_version_strips_v_prefix():
    assert parse_release_version({"tag_name": "v0.0.5"}) == "0.0.5"


def test_parse_release_version_rejects_bad_payloads():
    assert parse_release_version({}) is None
    assert parse_release_version({"tag_name": "not-a-tag"}) is None
    assert parse_release_version(None) is None
    assert parse_release_version("v1.2.3") is None


# --------------------------------------------------------------------------- #
# State application
# --------------------------------------------------------------------------- #


def test_apply_updates_state_and_reports_change():
    identity.state.update(
        {"user_agent": "fx/0.0.5", "protocol_version": "0.0.1", "specification_version": "4"}
    )
    changed = apply("fx/0.0.6", ("0.0.2", "5"))
    assert changed is True
    assert identity.state == {
        "user_agent": "fx/0.0.6",
        "protocol_version": "0.0.2",
        "specification_version": "5",
    }


def test_apply_is_idempotent():
    first = apply("fx/0.0.6", ("0.0.2", "5"))
    second = apply("fx/0.0.6", ("0.0.2", "5"))
    assert first is True and second is False


def test_apply_respects_pinned_user_agent(monkeypatch):
    monkeypatch.setattr(identity, "PINNED_USER_AGENT", "fx/manual")
    before = identity.state["user_agent"]
    changed = apply("fx/0.0.9", None)
    assert identity.state["user_agent"] == before
    assert changed is False


def test_apply_partial_update_keeps_other_fields():
    before_ua = identity.state["user_agent"]
    changed = apply(None, ("9.9", "8"))
    assert changed is True
    assert identity.state["user_agent"] == before_ua
    assert identity.state["protocol_version"] == "9.9"
    assert identity.state["specification_version"] == "8"


# --------------------------------------------------------------------------- #
# refresh(): end-to-end over a mocked transport
# --------------------------------------------------------------------------- #


def _run_refresh(handler) -> bool:
    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await refresh(client)

    return asyncio.run(go())


def test_refresh_applies_newer_upstream_identity():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.github.com":
            return httpx.Response(200, json={"tag_name": "v0.0.5"})
        return httpx.Response(200, text=RAW_SOURCE)

    assert _run_refresh(handler) is True
    assert identity.state["user_agent"] == "fx/0.0.5"
    assert identity.state["protocol_version"] == "0.0.2"
    assert identity.state["specification_version"] == "5"


def test_refresh_keeps_previous_state_on_http_error():
    identity.state.update(
        {"user_agent": "fx/0.0.5", "protocol_version": "0.0.1", "specification_version": "4"}
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    assert _run_refresh(handler) is False
    assert identity.state == {
        "user_agent": "fx/0.0.5",
        "protocol_version": "0.0.1",
        "specification_version": "4",
    }


def test_refresh_ignores_unparseable_source_but_keeps_version():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.github.com":
            return httpx.Response(200, json={"tag_name": "v0.0.6"})
        return httpx.Response(200, text="unexpected refactor")

    assert _run_refresh(handler) is True
    assert identity.state["user_agent"] == "fx/0.0.6"
    assert identity.state["protocol_version"] == identity.DEFAULT_PROTOCOL_VERSION


# --------------------------------------------------------------------------- #
# server integration: headers must track live state
# --------------------------------------------------------------------------- #


def test_server_headers_track_identity_state():
    from server import _v3_headers

    identity.state.update(
        {"user_agent": "fx/9.9.9", "protocol_version": "1.2", "specification_version": "7"}
    )
    headers = _v3_headers("zai/glm-5.2", streaming=True, api_key="test-gateway-key")
    assert headers["User-Agent"] == "fx/9.9.9"
    assert headers["ai-gateway-protocol-version"] == "1.2"
    assert headers["ai-language-model-specification-version"] == "7"


# --------------------------------------------------------------------------- #
# initialize(): live fetch before server starts
# --------------------------------------------------------------------------- #


def _run_initialize(handler, monkeypatch=None) -> None:
    """Run initialize() with a mocked transport and no local fx binary."""
    async def go():
        class FakeState:
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            await identity.initialize(FakeState())
        finally:
            await FakeState.client.aclose()
    if monkeypatch:
        monkeypatch.setattr(identity, "detect_local_fx_version", lambda: None)
    asyncio.run(go())


def test_initialize_fetches_live_version_from_github(monkeypatch):
    """initialize() populates user_agent from GitHub on first call."""
    identity.state["user_agent"] = None

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.github.com":
            return httpx.Response(200, json={"tag_name": "v0.0.7"})
        return httpx.Response(200, text=RAW_SOURCE)

    _run_initialize(handler, monkeypatch)
    assert identity.state["user_agent"] == "fx/0.0.7"
    assert identity.state["protocol_version"] == "0.0.2"
    assert identity.state["specification_version"] == "5"


def test_initialize_uses_local_binary_when_github_fails(monkeypatch):
    """If GitHub is unreachable, initialize falls back to local fx binary."""
    identity.state["user_agent"] = None
    monkeypatch.setattr(identity, "detect_local_fx_version", lambda: "0.0.5")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    _run_initialize(handler, monkeypatch)
    assert identity.state["user_agent"] == "fx/0.0.5"


def test_initialize_uses_fallback_when_all_sources_fail(monkeypatch):
    """If both GitHub and local binary fail, initialize uses the fallback."""
    identity.state["user_agent"] = None
    monkeypatch.setattr(identity, "detect_local_fx_version", lambda: None)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    _run_initialize(handler, monkeypatch)
    assert identity.state["user_agent"] == identity._FALLBACK_USER_AGENT
    assert identity.state["user_agent"].startswith("fx/")


def test_initialize_respects_pinned_user_agent(monkeypatch):
    """Pinned FX_USER_AGENT is never overwritten by initialize()."""
    monkeypatch.setattr(identity, "PINNED_USER_AGENT", "fx/pinned")
    monkeypatch.setattr(identity, "AUTO_UPDATE", False)
    monkeypatch.setattr(identity, "detect_local_fx_version", lambda: None)
    identity.state["user_agent"] = "fx/pinned"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"tag_name": "v9.9.9"})

    _run_initialize(handler, monkeypatch)
    assert identity.state["user_agent"] == "fx/pinned"


def test_initialize_never_leaves_user_agent_none(monkeypatch):
    """After initialize(), user_agent is always a non-None string."""
    identity.state["user_agent"] = None
    monkeypatch.setattr(identity, "detect_local_fx_version", lambda: None)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="total failure")

    _run_initialize(handler, monkeypatch)
    assert identity.state["user_agent"] is not None
    assert isinstance(identity.state["user_agent"], str)
    assert identity.state["user_agent"].startswith("fx/")
