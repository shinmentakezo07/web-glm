"""Tests for the SQLite-backed UsageStore."""

import os
import tempfile
import threading

from db import UsageStore


def _tmp_db() -> str:
    return tempfile.mktemp(suffix=".db")


def test_records_requests_errors_and_tokens():
    store = UsageStore(_tmp_db())
    store.record("1.2.3.4", model="m1", prompt_tokens=10, completion_tokens=5)
    store.record("1.2.3.4", model="m1", error=True)
    snap = store.snapshot()
    assert snap["enabled"] is True
    assert snap["totals"]["requests"] == 2
    assert snap["totals"]["errors"] == 1
    assert snap["totals"]["prompt_tokens"] == 10
    assert snap["totals"]["completion_tokens"] == 5
    assert snap["callers"]["1.2.3.4"]["requests"] == 2
    assert snap["models"]["m1"] == 2


def test_disabled_store_is_noop():
    store = UsageStore(_tmp_db(), enabled=False)
    store.record("x", prompt_tokens=99)
    assert store.snapshot()["totals"]["requests"] == 0


def test_negative_tokens_clamped():
    store = UsageStore(_tmp_db())
    store.record("x", prompt_tokens=-5, completion_tokens=3)
    t = store.totals()
    assert t["prompt_tokens"] == 0
    assert t["completion_tokens"] == 3


def test_totals_aggregate_across_callers():
    store = UsageStore(_tmp_db())
    store.record("a", model="m1")
    store.record("b", model="m2", prompt_tokens=7)
    totals = store.totals()
    assert totals["requests"] == 2
    assert totals["prompt_tokens"] == 7
    assert totals["cached_tokens"] == 0


def test_concurrent_records_are_thread_safe():
    store = UsageStore(_tmp_db())

    def worker():
        for _ in range(200):
            store.record("w", model="m")

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert store.snapshot()["totals"]["requests"] == 1600


def test_cached_and_reasoning_tokens_tracked():
    store = UsageStore(_tmp_db())
    store.record("caller1", model="m1",
                 prompt_tokens=100, completion_tokens=50,
                 cached_tokens=80, reasoning_tokens=10,
                 duration_ms=500.0)
    t = store.totals()
    assert t["cached_tokens"] == 80
    assert t["reasoning_tokens"] == 10


def test_tps_calculated_from_duration():
    store = UsageStore(_tmp_db())
    # 100 completion tokens over 2000 ms = 50 tok/s
    store.record("c", model="m", completion_tokens=100, duration_ms=2000.0)
    t = store.totals()
    assert t["tps"] == 50.0


def test_tps_zero_when_no_duration():
    store = UsageStore(_tmp_db())
    store.record("c", model="m", completion_tokens=100)
    t = store.totals()
    assert t["tps"] == 0


def test_persists_across_instances():
    """SQLite is the source of truth: data survives restart."""
    db = _tmp_db()
    if os.path.exists(db):
        os.unlink(db)
    store1 = UsageStore(db)
    store1.record("c", model="m", prompt_tokens=10, completion_tokens=5)
    store1._conn.close()
    store1._conn = None

    store2 = UsageStore(db)
    snap = store2.snapshot()
    assert snap["totals"]["requests"] == 1
    assert snap["totals"]["prompt_tokens"] == 10


def test_recent_requests():
    store = UsageStore(_tmp_db())
    store.record("c1", model="m1", prompt_tokens=10, completion_tokens=5, duration_ms=100)
    store.record("c2", model="m2", prompt_tokens=20, completion_tokens=10, duration_ms=200)
    recent = store.recent_requests(10)
    assert len(recent) == 2
    assert recent[0]["caller"] == "c2"  # most recent first
    assert recent[0]["tps"] == 50.0  # 10 tokens / 0.2 s
    assert recent[1]["caller"] == "c1"


def test_per_model_breakdown():
    store = UsageStore(_tmp_db())
    store.record("c", model="m1", prompt_tokens=10, completion_tokens=5)
    store.record("c", model="m1", prompt_tokens=20, completion_tokens=10)
    store.record("c", model="m2", prompt_tokens=5, completion_tokens=2)
    models = store.per_model()
    assert len(models) == 2
    m1 = [m for m in models if m["model"] == "m1"][0]
    assert m1["requests"] == 2
    assert m1["prompt_tokens"] == 30


def test_dashboard_payload():
    store = UsageStore(_tmp_db())
    store.record("c", model="m", prompt_tokens=10, completion_tokens=5, duration_ms=100)
    d = store.dashboard()
    assert "totals" in d
    assert "time_series" in d
    assert "per_model" in d
    assert "per_caller" in d
    assert "recent" in d
    assert d["totals"]["requests"] == 1


def test_time_series_buckets():
    store = UsageStore(_tmp_db())
    store.record("c", model="m", prompt_tokens=10, completion_tokens=5)
    ts = store.time_series()
    assert len(ts["labels"]) > 0
    assert sum(ts["prompt"]) == 10
    assert sum(ts["completion"]) == 5
