"""Multi-key support for the Vercel AI Gateway upstream.

Keys come from the environment:

    AI_GATEWAY_API_KEY      legacy single key (or comma-separated list)
    AI_GATEWAY_API_KEY_1    numbered keys; _1 wins over the legacy var on
    AI_GATEWAY_API_KEY_2    duplicates, order is legacy -> _1 -> _2 -> ...

Behaviour (all toggleable via .env):

    KEY_ROTATION=1    hand keys out round-robin when more than one is set
    KEY_FAILOVER=1    on a key-attributable error (401/402/403/408/429/5xx)
                      or a network error, transparently retry the next key
    KEY_COOLDOWN=30   seconds a failed key sits out of rotation (0 = off);
                      if every key is cooling, the coolest is still handed
                      out so the proxy degrades instead of dying
"""

from __future__ import annotations

import logging
import os
import threading
import time

log = logging.getLogger("gateway-proxy")

#: Upstream statuses that blame the key (or upstream capacity), not the
#: request body. 5xx is handled separately in should_failover().
_KEY_ATTRIBUTABLE = frozenset({401, 402, 403, 408, 429})

_MAX_NUMBERED_KEYS = 20


def _env_flag(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def mask(key: str) -> str:
    """Log-safe rendering of a key: only the tail is shown."""
    return f"…{key[-6:]}" if len(key) > 6 else "…"


def load_keys() -> list[str]:
    """Collect configured gateway keys in priority order, deduplicated."""
    keys: list[str] = []
    seen: set[str] = set()

    def add(raw: str | None) -> None:
        for part in (raw or "").split(","):
            key = part.strip()
            if key and key not in seen:
                seen.add(key)
                keys.append(key)

    add(os.getenv("AI_GATEWAY_API_KEY"))
    for i in range(1, _MAX_NUMBERED_KEYS + 1):
        add(os.getenv(f"AI_GATEWAY_API_KEY_{i}"))
    return keys


class KeyPool:
    """Round-robin key source with per-request failover and cooldowns.

    Thread-safe: route handlers run on the event loop, but the pool may
    also be probed from health checks or tests on other threads.
    """

    def __init__(
        self,
        keys: list[str],
        *,
        rotation: bool | None = None,
        failover: bool | None = None,
        cooldown_seconds: float | None = None,
    ):
        self.keys = list(keys)
        self.rotation = _env_flag("KEY_ROTATION") if rotation is None else rotation
        self.failover = _env_flag("KEY_FAILOVER") if failover is None else failover
        self.cooldown_seconds = (
            float(os.getenv("KEY_COOLDOWN", "30")) if cooldown_seconds is None else cooldown_seconds
        )
        self._idx = 0
        self._lock = threading.Lock()
        self._cooldown_until: dict[str, float] = {}
        self._failures: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self.keys)

    def next(self, exclude: set[str] | frozenset[str] = frozenset()) -> str | None:
        """Pick the next key. `exclude` holds keys already tried for the
        current request; returns None when every key is excluded."""
        now = time.monotonic()
        with self._lock:
            n = len(self.keys)
            if n == 0:
                return None
            start = self._idx if self.rotation else 0
            order = [(start + off) % n for off in range(n)]
            pick = -1
            # Pass 1: skip cooling keys. Pass 2: accept them (all cooling
            # must not take the proxy down).
            for allow_cooling in (False, True):
                for i in order:
                    key = self.keys[i]
                    if key in exclude:
                        continue
                    if not allow_cooling and self._cooldown_until.get(key, 0.0) > now:
                        continue
                    pick = i
                    break
                if pick >= 0:
                    break
            if pick < 0:
                return None
            if self.rotation:
                self._idx = (pick + 1) % n
            return self.keys[pick]

    def should_failover(self, status_code: int) -> bool:
        """True when this upstream status justifies retrying on another key."""
        return self.failover and (status_code in _KEY_ATTRIBUTABLE or status_code >= 500)

    def report_failure(self, key: str) -> None:
        """Put a key on cooldown and bump its failure counter."""
        with self._lock:
            self._failures[key] = self._failures.get(key, 0) + 1
            if self.cooldown_seconds > 0:
                self._cooldown_until[key] = time.monotonic() + self.cooldown_seconds

    def report_success(self, key: str) -> None:
        """A working key leaves cooldown immediately."""
        with self._lock:
            self._cooldown_until.pop(key, None)

    def stats(self) -> dict:
        """Non-sensitive snapshot for /healthz."""
        now = time.monotonic()
        with self._lock:
            return {
                "count": len(self.keys),
                "rotation": self.rotation,
                "failover": self.failover,
                "cooldown_s": self.cooldown_seconds,
                "cooling": sum(1 for until in self._cooldown_until.values() if until > now),
            }
