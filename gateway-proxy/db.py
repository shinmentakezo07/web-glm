"""SQLite-backed usage accounting for proxy traffic.

Replaces the in-memory ``UsageTracker`` as the single source of truth.
Every request is written as a row in ``requests``; aggregates are computed
on the fly via SQL. The database survives restarts.

Thread-safe: writes are serialised through a lock; reads use the same
connection with ``check_same_thread=False``.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time


_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL    NOT NULL,
    caller      TEXT    NOT NULL DEFAULT 'unknown',
    model       TEXT    NOT NULL DEFAULT '',
    endpoint    TEXT    NOT NULL DEFAULT '',
    error       INTEGER NOT NULL DEFAULT 0,
    prompt_tokens       INTEGER NOT NULL DEFAULT 0,
    completion_tokens   INTEGER NOT NULL DEFAULT 0,
    cached_tokens       INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens    INTEGER NOT NULL DEFAULT 0,
    duration_ms REAL    NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_requests_ts     ON requests(ts);
CREATE INDEX IF NOT EXISTS idx_requests_model  ON requests(model);
CREATE INDEX IF NOT EXISTS idx_requests_caller  ON requests(caller);
"""


class UsageStore:
    """SQLite-persisted, thread-safe request/token/TPS store.

    The database file survives restarts. By default it lives next to the
    module file (``usage.db``) so the same DB is used regardless of the
    current working directory.
    """

    # Default path is next to this module, NOT cwd — so restarting from
    # a different directory doesn't create a fresh DB.
    _DEFAULT_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "usage.db")

    def __init__(self, db_path: str | None = None, *, enabled: bool = True):
        self.enabled = enabled
        self._db_path = db_path or os.getenv("USAGE_DB_PATH", self._DEFAULT_DB)
        self._lock = threading.Lock()
        self._started = time.time()
        self._data_start = self._started
        self._conn: sqlite3.Connection | None = None
        if self.enabled:
            self._init_db()
            # Track the earliest record so uptime reflects data age, not
            # just the current process lifetime.
            self._data_start = self._earliest_ts()

    # -- internal ----------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                self._db_path, check_same_thread=False, isolation_level=None
            )
            self._conn.row_factory = sqlite3.Row
            # WAL mode allows concurrent readers without blocking writers;
            # a busy timeout lets us wait briefly instead of erroring
            # when another connection holds the lock.
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.executescript(_SCHEMA)
        return self._conn

    def _reconnect(self) -> sqlite3.Connection:
        """Close the stale connection and open a fresh one."""
        try:
            if self._conn is not None:
                self._conn.close()
        except Exception:
            pass
        self._conn = None
        return self._connect()

    def _init_db(self) -> None:
        with self._lock:
            self._connect()

    def _earliest_ts(self) -> float:
        """Return the earliest record timestamp, or process start if empty."""
        try:
            with self._lock:
                conn = self._connect()
                row = conn.execute("SELECT MIN(ts) FROM requests").fetchone()
            if row and row[0] is not None:
                return float(row[0])
        except Exception:
            pass
        return self._started

    # -- public API --------------------------------------------------------

    def record(
        self,
        caller: str,
        *,
        model: str | None = None,
        endpoint: str = "",
        error: bool = False,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cached_tokens: int = 0,
        reasoning_tokens: int = 0,
        duration_ms: float = 0.0,
    ) -> None:
        if not self.enabled:
            return
        now = time.time()
        with self._lock:
            for attempt in range(2):
                try:
                    conn = self._connect() if attempt == 0 else self._reconnect()
                    conn.execute(
                        """INSERT INTO requests
                               (ts, caller, model, endpoint, error,
                                prompt_tokens, completion_tokens, cached_tokens,
                                reasoning_tokens, duration_ms)
                           VALUES (?,?,?,?,?,?,?,?,?,?)""",
                        (
                            now,
                            caller,
                            model or "",
                            endpoint,
                            1 if error else 0,
                            max(0, prompt_tokens),
                            max(0, completion_tokens),
                            max(0, cached_tokens),
                            max(0, reasoning_tokens),
                            max(0.0, duration_ms),
                        ),
                    )
                    break
                except (sqlite3.OperationalError, sqlite3.DatabaseError):
                    if attempt == 0:
                        continue
                    # Usage tracking is a side-effect — never crash a request
                    # because the DB is locked, readonly, or unavailable.
                    pass

    # -- queries -----------------------------------------------------------

    def _query(self, sql: str, params: tuple = ()) -> list[dict]:
        try:
            with self._lock:
                conn = self._connect()
                rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]
        except (sqlite3.OperationalError, sqlite3.DatabaseError):
            return []

    def _query_one(self, sql: str, params: tuple = ()) -> dict:
        try:
            with self._lock:
                conn = self._connect()
                row = conn.execute(sql, params).fetchone()
            return dict(row) if row else {}
        except (sqlite3.OperationalError, sqlite3.DatabaseError):
            return {}

    def totals(self) -> dict:
        """Aggregated snapshot across all callers (for /healthz)."""
        row = self._query_one(
            """SELECT
                   COUNT(*)                   AS requests,
                   SUM(error)                  AS errors,
                   COALESCE(SUM(prompt_tokens), 0)      AS prompt_tokens,
                   COALESCE(SUM(completion_tokens), 0)  AS completion_tokens,
                   COALESCE(SUM(cached_tokens), 0)       AS cached_tokens,
                   COALESCE(SUM(reasoning_tokens), 0)    AS reasoning_tokens,
                   COALESCE(SUM(duration_ms), 0)         AS total_duration_ms
               FROM requests"""
        )
        total_duration_s = (row.get("total_duration_ms") or 0) / 1000.0
        completion = row.get("completion_tokens") or 0
        return {
            "uptime_s": round(time.time() - self._data_start),
            "requests": row.get("requests") or 0,
            "errors": row.get("errors") or 0,
            "prompt_tokens": row.get("prompt_tokens") or 0,
            "completion_tokens": completion,
            "cached_tokens": row.get("cached_tokens") or 0,
            "reasoning_tokens": row.get("reasoning_tokens") or 0,
            "tps": round(completion / total_duration_s, 2) if total_duration_s > 0 else 0,
        }

    def snapshot(self) -> dict:
        """Full per-caller breakdown (for GET /v1/usage)."""
        callers = self._query(
            """SELECT caller,
                      COUNT(*)                   AS requests,
                      SUM(error)                  AS errors,
                      COALESCE(SUM(prompt_tokens), 0)      AS prompt_tokens,
                      COALESCE(SUM(completion_tokens), 0)  AS completion_tokens,
                      COALESCE(SUM(cached_tokens), 0)       AS cached_tokens,
                      COALESCE(SUM(reasoning_tokens), 0)    AS reasoning_tokens,
                      MAX(ts)                     AS last_request_ts
               FROM requests GROUP BY caller"""
        )
        for c in callers:
            c["last_request_ts"] = c.get("last_request_ts") or 0.0
        models = self._query(
            """SELECT model, COUNT(*) AS count
               FROM requests WHERE model != '' GROUP BY model ORDER BY count DESC"""
        )
        return {
            "enabled": self.enabled,
            "uptime_s": round(time.time() - self._data_start),
            "totals": self.totals(),
            "callers": {c["caller"]: c for c in callers},
            "models": {m["model"]: m["count"] for m in models},
        }

    # -- dashboard-specific queries ----------------------------------------

    def recent_requests(self, limit: int = 20) -> list[dict]:
        rows = self._query(
            """SELECT ts, caller, model, endpoint, error,
                      prompt_tokens, completion_tokens, cached_tokens,
                      reasoning_tokens, duration_ms
               FROM requests ORDER BY id DESC LIMIT ?""",
            (limit,),
        )
        for r in rows:
            # Guard against sub-millisecond durations producing absurd TPS
            # (e.g. cached responses that return in under 1ms).
            dur_s = max(r["duration_ms"] or 0, 1.0) / 1000.0
            r["tps"] = (
                round(r["completion_tokens"] / dur_s, 2)
                if r["completion_tokens"]
                else 0
            )
            r["error"] = bool(r["error"])
        return rows

    def time_series(self, since: float | None = None, buckets: int = 60) -> dict:
        """Token throughput bucketed over the last hour (or since *since*).

        Returns ``{"labels": [...], "prompt": [...], "completion": [...],
        "cached": [...], "requests": [...]}``.
        """
        if since is None:
            since = time.time() - 3600
        rows = self._query(
            """SELECT ts, prompt_tokens, completion_tokens, cached_tokens
               FROM requests WHERE ts >= ? ORDER BY id""",
            (since,),
        )
        if not rows:
            return {"labels": [], "prompt": [], "completion": [], "cached": [], "requests": []}

        now = time.time()
        window = now - since
        bucket_size = max(1.0, window / buckets)

        labels: list[str] = []
        p_vals: list[int] = []
        c_vals: list[int] = []
        cache_vals: list[int] = []
        r_vals: list[int] = []

        idx = 0
        for b in range(buckets):
            b_start = since + b * bucket_size
            b_end = b_start + bucket_size
            p = c = cache = cnt = 0
            while idx < len(rows) and rows[idx]["ts"] < b_end:
                p += rows[idx]["prompt_tokens"]
                c += rows[idx]["completion_tokens"]
                cache += rows[idx]["cached_tokens"]
                cnt += 1
                idx += 1
            labels.append(time.strftime("%H:%M", time.localtime(b_start)))
            p_vals.append(p)
            c_vals.append(c)
            cache_vals.append(cache)
            r_vals.append(cnt)

        return {
            "labels": labels,
            "prompt": p_vals,
            "completion": c_vals,
            "cached": cache_vals,
            "requests": r_vals,
        }

    def per_model(self) -> list[dict]:
        return self._query(
            """SELECT model,
                      COUNT(*)                   AS requests,
                      SUM(error)                  AS errors,
                      COALESCE(SUM(prompt_tokens), 0)      AS prompt_tokens,
                      COALESCE(SUM(completion_tokens), 0)  AS completion_tokens,
                      COALESCE(SUM(cached_tokens), 0)       AS cached_tokens,
                      COALESCE(SUM(reasoning_tokens), 0)    AS reasoning_tokens
               FROM requests WHERE model != ''
               GROUP BY model ORDER BY requests DESC"""
        )

    def per_caller(self) -> list[dict]:
        return self._query(
            """SELECT caller,
                      COUNT(*)                   AS requests,
                      SUM(error)                  AS errors,
                      COALESCE(SUM(prompt_tokens), 0)      AS prompt_tokens,
                      COALESCE(SUM(completion_tokens), 0)  AS completion_tokens,
                      COALESCE(SUM(cached_tokens), 0)       AS cached_tokens,
                      MAX(ts)                     AS last_request_ts
               FROM requests GROUP BY caller ORDER BY requests DESC"""
        )

    def dashboard(self) -> dict:
        """Single payload for the dashboard: totals + series + breakdowns."""
        return {
            "totals": self.totals(),
            "time_series": self.time_series(),
            "per_model": self.per_model(),
            "per_caller": self.per_caller(),
            "recent": self.recent_requests(20),
        }
