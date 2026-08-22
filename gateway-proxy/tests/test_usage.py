import threading

from usage import UsageTracker


def test_records_requests_errors_and_tokens():
    t = UsageTracker()
    t.record("1.2.3.4", model="m1", prompt_tokens=10, completion_tokens=5)
    t.record("1.2.3.4", model="m1", error=True)
    snap = t.snapshot()
    assert snap["enabled"] is True
    assert snap["totals"]["requests"] == 2
    assert snap["totals"]["errors"] == 1
    assert snap["totals"]["prompt_tokens"] == 10
    assert snap["totals"]["completion_tokens"] == 5
    assert snap["callers"]["1.2.3.4"]["models"]["m1"] == 2


def test_disabled_tracker_is_noop():
    t = UsageTracker(enabled=False)
    t.record("x", prompt_tokens=99)
    assert t.snapshot()["totals"]["requests"] == 0


def test_negative_tokens_clamped():
    t = UsageTracker()
    t.record("x", prompt_tokens=-5, completion_tokens=3)
    assert t.snapshot()["totals"]["prompt_tokens"] == 0
    assert t.snapshot()["totals"]["completion_tokens"] == 3


def test_totals_aggregate_across_callers():
    t = UsageTracker()
    t.record("a", model="m1")
    t.record("b", model="m2", prompt_tokens=7)
    totals = t.totals()
    assert totals["requests"] == 2
    assert totals["prompt_tokens"] == 7
    assert totals["models"] == {"m1": 1, "m2": 1}


def test_concurrent_records_are_thread_safe():
    t = UsageTracker()

    def worker():
        for _ in range(200):
            t.record("w", model="m")

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert t.snapshot()["totals"]["requests"] == 1600