"""Live fx CLI identity sync.

The proxy mimics the fx binary on the wire. If Vercel ships a new fx release
(new ``fx/<version>`` User-Agent) or bumps the protocol headers, a stale
hardcoded identity may get us rejected. This module keeps the identity fresh
by fetching, on a background loop:

  - latest release tag:  https://api.github.com/repos/vercel-labs/fx/releases/latest
                         -> User-Agent becomes ``fx/<tag without v>``
  - raw gateway source:  https://raw.githubusercontent.com/vercel-labs/fx/main/src/gateway/client.zig
                         -> regex out ai-gateway-protocol-version and
                            ai-language-model-specification-version

State hot-swaps in memory; every upstream request reads current values.
Env knobs:
  FX_AUTO_UPDATE=0        disable background sync (keep defaults)
  FX_USER_AGENT=fx/1.2.3  pin the User-Agent manually (never overwritten)
  FX_REFRESH_SECS=3600    refresh interval
  FX_RELEASES_URL=...     override releases API endpoint
  FX_SOURCE_RAW_URL=...   override raw source URL
"""

from __future__ import annotations

import asyncio
import logging
import os
import re

import httpx

log = logging.getLogger("gateway-proxy.identity")

RELEASES_URL = os.getenv(
    "FX_RELEASES_URL", "https://api.github.com/repos/vercel-labs/fx/releases/latest"
)
SOURCE_RAW_URL = os.getenv(
    "FX_SOURCE_RAW_URL",
    "https://raw.githubusercontent.com/vercel-labs/fx/main/src/gateway/client.zig",
)
REFRESH_INTERVAL = float(os.getenv("FX_REFRESH_SECS", "3600"))
AUTO_UPDATE = os.getenv("FX_AUTO_UPDATE", "1").lower() in ("1", "true", "yes")
PINNED_USER_AGENT = os.getenv("FX_USER_AGENT", "")

DEFAULT_PROTOCOL_VERSION = "0.0.1"
DEFAULT_SPEC_VERSION = "4"

# Single mutable source of truth; server.py reads it via _v3_headers() and
# openai_to_v3(product_user_agent=...) on every request.
state: dict[str, str] = {
    "user_agent": PINNED_USER_AGENT or "fx/0.0.4",
    "protocol_version": DEFAULT_PROTOCOL_VERSION,
    "specification_version": DEFAULT_SPEC_VERSION,
}

_PROTOCOL_RE = re.compile(r'"ai-gateway-protocol-version",\s*\.value\s*=\s*"([^"]+)"')
_SPEC_RE = re.compile(r'"ai-language-model-specification-version",\s*\.value\s*=\s*"([^"]+)"')


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def parse_source_versions(text: str) -> tuple[str, str] | None:
    """Extract (protocol_version, specification_version) from raw fx Zig source."""
    proto = _PROTOCOL_RE.search(text)
    spec = _SPEC_RE.search(text)
    if proto is None or spec is None:
        return None
    return proto.group(1), spec.group(1)


def parse_release_version(payload: object) -> str | None:
    """Extract the released fx version from a releases API body ('v0.0.5' -> '0.0.5')."""
    if not isinstance(payload, dict):
        return None
    tag = payload.get("tag_name")
    if not isinstance(tag, str) or not tag.startswith("v") or len(tag) < 2:
        return None
    return tag[1:]


# --------------------------------------------------------------------------- #
# Fetch + apply
# --------------------------------------------------------------------------- #


async def fetch_latest_version(client: httpx.AsyncClient) -> str | None:
    resp = await client.get(RELEASES_URL, headers={"Accept": "application/vnd.github+json"})
    resp.raise_for_status()
    return parse_release_version(resp.json())


async def fetch_source_versions(client: httpx.AsyncClient) -> tuple[str, str] | None:
    resp = await client.get(SOURCE_RAW_URL)
    resp.raise_for_status()
    return parse_source_versions(resp.text)


def apply(user_agent: str | None, versions: tuple[str, str] | None) -> bool:
    """Merge fetched values into state. Returns True when anything changed."""
    changed = False
    if user_agent and not PINNED_USER_AGENT and user_agent != state["user_agent"]:
        log.info("fx user-agent updated: %s -> %s", state["user_agent"], user_agent)
        state["user_agent"] = user_agent
        changed = True
    if versions and versions != (state["protocol_version"], state["specification_version"]):
        log.info(
            "fx protocol headers updated: %s -> %s",
            (state["protocol_version"], state["specification_version"]),
            versions,
        )
        state["protocol_version"], state["specification_version"] = versions
        changed = True
    return changed


async def refresh(client: httpx.AsyncClient) -> bool:
    """Fetch current fx identity from GitHub; apply changes. Never raises."""
    try:
        version = await fetch_latest_version(client)
        versions = await fetch_source_versions(client)
    except Exception as exc:  # noqa: BLE001 — identity sync must never break the proxy
        log.warning("fx identity refresh failed (%s); keeping %s", exc, state)
        return False
    return apply(f"fx/{version}" if version else None, versions)


# --------------------------------------------------------------------------- #
# Local-binary fallback
# --------------------------------------------------------------------------- #


def detect_local_fx_version() -> str | None:
    """Best-effort: ask a locally installed fx CLI for its version.

    Used as a fallback identity source when the GitHub refresh is disabled
    or unreachable, so the User-Agent never goes stale relative to the fx
    build actually present on this machine (e.g. repo default ``fx/0.0.4``
    while ``fx/0.0.5`` is installed).
    """
    import shutil
    import subprocess

    exe = shutil.which("fx")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--version"], capture_output=True, text=True, timeout=10
        )
    except Exception:  # noqa: BLE001 — never let detection break startup
        return None
    lines = (out.stdout or "").strip().splitlines()
    if not lines:
        return None
    candidate = lines[0].strip()
    if candidate.startswith("v"):
        candidate = candidate[1:]
    return candidate or None


# --------------------------------------------------------------------------- #
# Background loop
# --------------------------------------------------------------------------- #


async def refresher_loop(client: httpx.AsyncClient, interval: float) -> None:
    while True:
        await refresh(client)
        await asyncio.sleep(interval)


def start(app_state: object) -> asyncio.Task | None:
    """Start the background refresher sharing the app's httpx client.

    Returns the task (store it to cancel at shutdown), or None when disabled.
    """
    # Local-binary fallback runs even with FX_AUTO_UPDATE=0: prefer the
    # installed fx version over the hardcoded default so the wire identity
    # stays fresh offline. GitHub refresh remains the primary source.
    local_version = detect_local_fx_version()
    if (
        local_version
        and not PINNED_USER_AGENT
        and f"fx/{local_version}" != state["user_agent"]
    ):
        log.info("using local fx binary version for identity: fx/%s", local_version)
        apply(f"fx/{local_version}", None)
    if not AUTO_UPDATE:
        log.info("fx identity auto-update disabled")
        return None
    client: httpx.AsyncClient | None = getattr(app_state, "client", None)
    if client is None:
        return None
    task = asyncio.create_task(refresher_loop(client, REFRESH_INTERVAL))
    task.add_done_callback(_log_unexpected_exit)
    return task


def _log_unexpected_exit(task: asyncio.Task) -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        log.error("fx identity refresher died unexpectedly: %r", exc)
