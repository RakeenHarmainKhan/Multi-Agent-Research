from datetime import datetime, timezone

from app.core.runtime_store import RuntimeStore


def _result(thread_id="t1", status="completed", execution_time=2.0, **extra):
    return {
        "thread_id": thread_id,
        "query": "q",
        "status": status,
        "execution_time": execution_time,
        "created_at": datetime.now(timezone.utc),
        **extra,
    }


def test_save_and_get_roundtrip():
    store = RuntimeStore()
    store.save_result(_result("a"))
    got = store.get_result("a")
    assert got["thread_id"] == "a"
    assert "updated_at" in got


def test_get_unknown_returns_none():
    assert RuntimeStore().get_result("missing") is None


def test_get_returns_copy():
    store = RuntimeStore()
    store.save_result(_result("a"))
    store.get_result("a")["status"] = "mutated"
    assert store.get_result("a")["status"] == "completed"


def test_history_is_newest_first_and_deduplicated():
    store = RuntimeStore()
    store.save_result(_result("a"))
    store.save_result(_result("b"))
    store.save_result(_result("a", status="failed"))  # update, not a new entry
    ids = [item["thread_id"] for item in store.list_history()]
    assert ids == ["b", "a"]
    assert store.get_result("a")["status"] == "failed"


def test_history_limit():
    store = RuntimeStore()
    for i in range(5):
        store.save_result(_result(f"t{i}"))
    assert len(store.list_history(limit=2)) == 2


def test_stats_empty():
    stats = RuntimeStore().stats()
    assert stats["total_queries"] == 0
    assert stats["avg_latency"] == 0.0


def test_stats_counts_and_latency():
    store = RuntimeStore()
    store.save_result(_result("a", "completed", 2.0))
    store.save_result(_result("b", "completed", 4.0))
    store.save_result(_result("c", "failed", 9.0))  # failed runs excluded from latency
    stats = store.stats()
    assert stats["total_queries"] == 3
    assert stats["successful_queries"] == 2
    assert stats["failed_queries"] == 1
    assert stats["avg_latency"] == 3.0
    assert stats["queries_today"] == 3
