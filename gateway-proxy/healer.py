"""Key healer: probe cooling keys and restore healthy ones early.

Failover puts a failed key on a KEY_COOLDOWN (default 30s) during which the
pool skips it. If the upstream hiccup was transient — capacity blip, brief
429 burst — the key is actually fine again seconds later, but with a single
key pool the proxy degrades until the cooldown expires.

The healer closes that gap: on a background loop it sweeps every cooling key
and sends a cheap authenticated GET /v1/models probe. A 200 clears the
cooldown immediately (report_success), making the key available again; any
other status or network error leaves the cooldown to expire naturally. The
sweep never raises and never touches non-cooling keys.

Env knobs:

    KEY_HEALER=0           disable the background sweep (default on)
    KEY_HEAL_INTERVAL=15   seconds between sweeps
"""

from __future__ import annotations

import asyncio
import logging
import os

import httpx

from keys import KeyPool, mask

log = logging.getLogger("gateway-proxy.healer")

ENABLED = os.getenv("KEY_HEALER", "1").strip().lower() in ("1", "true", "yes", "on")
HEAL_INTERVAL = float(os.getenv("KEY_HEAL_INTERVAL", "15"))


async def heal_once(pool: KeyPool, probe) -> int:
    """Probe every cooling key once; clear cooldowns of healthy keys.

    Returns the number of keys healed. A probe raising anything counts as
    unhealthy — probing must never break the loop.
    """
    healed = 0
    for key in pool.cooling_keys():
        try:
            ok = await probe(key)
        except Exception:  # noqa: BLE001 — a broken probe must not kill healing
            ok = False
        if ok:
            pool.report_success(key)
            log.info("gateway key %s healed; available again", mask(key))
            healed += 1
    return healed


def make_probe(client: httpx.AsyncClient, url: str):
    """Build an async probe(key) -> bool against a cheap upstream endpoint."""

    async def probe(key: str) -> bool:
        try:
            resp = await client.get(
                url,
                headers={"Accept": "application/json", "Authorization": f"Bearer {key}"},
            )
        except httpx.RequestError:
            return False
        return resp.status_code == 200

    return probe


async def heal_loop(
    pool: KeyPool, models_url: str, interval: float
) -> None:
    """Sweep cooling keys every `interval` seconds. Never raises.

    Uses a dedicated httpx client so healer probes don't compete with real
    requests for connection-pool slots on the shared app client.
    """
    async with httpx.AsyncClient(timeout=15.0) as client:
        probe = make_probe(client, models_url)
        while True:
            await asyncio.sleep(interval)
            try:
                await heal_once(pool, probe)
            except Exception as exc:  # noqa: BLE001 — the loop must survive bugs
                log.warning("key-healer sweep failed unexpectedly: %r", exc)


def start(
    app_state: object, *, pool: KeyPool, models_url: str
) -> asyncio.Task | None:
    """Start the background healer with its own httpx client.

    Returns the task (cancel at shutdown), or None when disabled or there
    is nothing to heal.
    """
    if not ENABLED:
        log.info("key healer disabled (KEY_HEALER=0)")
        return None
    if not len(pool):
        return None
    log.info("key healer: sweeping cooling keys every %.0fs via %s", HEAL_INTERVAL, models_url)
    task = asyncio.create_task(heal_loop(pool, models_url, HEAL_INTERVAL))
    task.add_done_callback(_log_unexpected_exit)
    return task


def _log_unexpected_exit(task: asyncio.Task) -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        log.error("key healer died unexpectedly: %r", exc)
