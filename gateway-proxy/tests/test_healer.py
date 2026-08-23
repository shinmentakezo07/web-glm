"""Tests for the key healer (healer module).

The healer sweeps keys that are serving a cooldown and probes the upstream
Gateway with a cheap authenticated request. A key that answers 200 leaves
cooldown immediately (report_success) instead of sitting out the full
KEY_COOLDOWN window, so a recovered key is available again within seconds.
"""

import asyncio
import importlib

import httpx
import pytest

import healer
from healer import heal_once, make_probe, start
from keys import KeyPool


def run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# heal_once
# --------------------------------------------------------------------------- #


class TestHealOnce:
    def test_healthy_key_leaves_cooldown(self):
        pool = KeyPool(["k1", "k2"], cooldown_seconds=60.0)
        pool.report_failure("k1")

        async def probe(key: str) -> bool:
            return True

        assert run(heal_once(pool, probe)) == 1
        assert pool.cooling_keys() == []

    def test_sick_key_stays_cooling(self):
        pool = KeyPool(["k1", "k2"], cooldown_seconds=60.0)
        pool.report_failure("k1")
        pool.report_failure("k2")

        async def probe(key: str) -> bool:
            return key == "k1"

        assert run(heal_once(pool, probe)) == 1
        assert pool.cooling_keys() == ["k2"]

    def test_no_cooling_keys_means_no_probes(self):
        pool = KeyPool(["k1"], cooldown_seconds=60.0)
        probed: list[str] = []

        async def probe(key: str) -> bool:
            probed.append(key)
            return True

        assert run(heal_once(pool, probe)) == 0
        assert probed == []

    def test_probe_exception_counts_as_unhealthy(self):
        pool = KeyPool(["k1"], cooldown_seconds=60.0)
        pool.report_failure("k1")

        async def probe(key: str) -> bool:
            raise RuntimeError("boom")

        assert run(heal_once(pool, probe)) == 0
        assert pool.cooling_keys() == ["k1"]


# --------------------------------------------------------------------------- #
# make_probe against a mocked transport
# --------------------------------------------------------------------------- #


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class TestMakeProbe:
    def test_200_is_healthy(self):
        async def main():
            async with _client(
                lambda request: httpx.Response(200, json={"data": []})
            ) as client:
                probe = make_probe(client, "https://gw.test/v1/models")
                return await probe("k1")

        assert run(main()) is True

    def test_non_200_is_unhealthy(self):
        async def main():
            async with _client(lambda request: httpx.Response(401)) as client:
                probe = make_probe(client, "https://gw.test/v1/models")
                return await probe("k1")

        assert run(main()) is False

    def test_network_error_is_unhealthy(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("down")

        async def main():
            async with _client(handler) as client:
                probe = make_probe(client, "https://gw.test/v1/models")
                return await probe("k1")

        assert run(main()) is False

    def test_probe_sends_bearer_key(self):
        seen: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["auth"] = request.headers.get("Authorization", "")
            return httpx.Response(200)

        async def main():
            async with _client(handler) as client:
                probe = make_probe(client, "https://gw.test/v1/models")
                await probe("secret-key")

        run(main())
        assert seen["auth"] == "Bearer secret-key"


# --------------------------------------------------------------------------- #
# start() gating + loop lifecycle
# --------------------------------------------------------------------------- #


class _AppState:
    client = object()


class TestStart:
    def test_disabled_by_flag(self, monkeypatch):
        monkeypatch.setattr(healer, "ENABLED", False)
        assert (
            start(_AppState(), pool=KeyPool(["k1"]), models_url="https://gw.test/v1/models")
            is None
        )

    def test_no_client_no_task(self):
        assert (
            start(object(), pool=KeyPool(["k1"]), models_url="https://gw.test/v1/models") is None
        )

    def test_empty_pool_no_task(self):
        assert (
            start(_AppState(), pool=KeyPool([]), models_url="https://gw.test/v1/models") is None
        )

    def test_starts_and_cancels(self):
        async def main():
            task = start(
                _AppState(), pool=KeyPool(["k1"]), models_url="https://gw.test/v1/models"
            )
            assert task is not None
            await asyncio.sleep(0)
            assert not task.done()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        run(main())


def test_env_defaults(monkeypatch):
    monkeypatch.setenv("KEY_HEALER", "1")
    monkeypatch.setenv("KEY_HEAL_INTERVAL", "7")
    h = importlib.reload(healer)
    try:
        assert h.ENABLED is True
        assert h.HEAL_INTERVAL == 7.0
    finally:
        monkeypatch.setenv("KEY_HEAL_INTERVAL", "15")
        importlib.reload(h)
