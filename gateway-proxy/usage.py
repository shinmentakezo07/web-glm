"""In-memory usage accounting for proxy traffic.

Counters live in process memory (reset on restart) and are keyed by the
calling client's host. Token counts come from whatever the upstream reports
in its usage field; streams that end without a usage chunk still count as
requests. Set USAGE_TRACKING=0 to disable (record becomes a no-op).

Thread-safe: route handlers run on the event loop but /healthz probes may
read from other threads.
"""

from __future__ import annotations

import threading
import time


class UsageTracker:
    """Aggregate request/error/token counters per caller."""

    def __init__(self, *, enabled: bool = True):
        self.enabled = enabled
        self._lock = threading.Lock()
        self._started = time.time()
        self._callers: dict[str, dict] = {}

    def record(
        self,
        caller: str,
        *,
        model: str | None = None,
        error: bool = False,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        if not self.enabled:
            return
        now = time.time()
        with self._lock:
            entry = self._callers.setdefault(caller, {
                "requests": 0,
                "errors": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "last_request_ts": 0.0,
                "models": {},
            })
            entry["requests"] += 1
            if error:
                entry["errors"] += 1
            entry["prompt_tokens"] += max(0, prompt_tokens)
            entry["completion_tokens"] += max(0, completion_tokens)
            entry["last_request_ts"] = now
            if model:
                entry["models"][model] = entry["models"].get(model, 0) + 1

    # Internal: caller must hold the lock.
    def _totals_unlocked(self) -> dict:
        totals = {
            "requests": 0,
            "errors": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "models": {},
        }
        for entry in self._callers.values():
            for key in ("requests", "errors", "prompt_tokens", "completion_tokens"):
                totals[key] += entry[key]
            for model, count in entry["models"].items():
                totals["models"][model] = totals["models"].get(model, 0) + count
        return totals

    def totals(self) -> dict:
        """Aggregated snapshot across all callers (for /healthz)."""
        with self._lock:
            return {
                "uptime_s": round(time.time() - self._started),
                **self._totals_unlocked(),
            }

    def snapshot(self) -> dict:
        """Full per-caller breakdown (for GET /v1/usage)."""
        with self._lock:
            return {
                "enabled": self.enabled,
                "uptime_s": round(time.time() - self._started),
                "totals": self._totals_unlocked(),
                "callers": {caller: dict(entry) for caller, entry in self._callers.items()},
            }