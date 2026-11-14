from datetime import datetime, timezone
from itertools import count

from app.core.runtime_store import RuntimeStore

_ids = count()


def _result(thread_id="t1", status="completed", execution_time=2.0, run_id=None, **extra):
    return {
        "run_id": run_id or f"run-{next(_ids)}",
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


def test_each_question_in_a_conversation_is_its_own_run():
    """Regression: runs used to be keyed by thread_id, so a conversation counted as one query."""
    store = RuntimeStore()
    store.save_result(_result("chat-1", query="first"))
    store.save_result(_result("chat-1", query="second"))
    assert [h["query"] for h in store.list_history()] == ["second", "first"]
    assert store.stats()["total_queries"] == 2
    assert store.get_result("chat-1")["query"] == "second"  # latest run for the thread


def test_saving_same_run_id_updates_in_place():
    store = RuntimeStore()
    store.save_result(_result("a", run_id="r1"))
    store.save_result(_result("a", run_id="r1", status="failed"))
    assert len(store.list_history()) == 1
    assert store.get_result("a")["status"] == "failed"


def test_history_is_newest_first():
    store = RuntimeStore()
    store.save_result(_result("a"))
    store.save_result(_result("b"))
    assert [item["thread_id"] for item in store.list_history()] == ["b", "a"]


def test_history_limit():
    store = RuntimeStore()
    for i in range(5):
        store.save_result(_result(f"t{i}"))
    assert len(store.list_history(limit=2)) == 2


def test_oldest_runs_are_evicted_at_capacity(monkeypatch):
    from collections import deque

    store = RuntimeStore()
    store._history = deque(maxlen=2)
    for i in range(3):
        store.save_result(_result(f"t{i}"))
    assert store.get_result("t0") is None
    assert store.stats()["total_queries"] == 2


def test_clear():
    store = RuntimeStore()
    store.save_result(_result("a"))
    store.clear()
    assert store.get_result("a") is None and store.list_history() == []


def test_stats_empty():
    stats = RuntimeStore().stats()
    assert stats["total_queries"] == 0
    assert stats["avg_latency"] == 0.0
    assert stats["total_tokens"] == 0
    assert stats["total_cost"] is None


def test_stats_counts_latency_and_tokens():
    store = RuntimeStore()
    store.save_result(_result("a", "completed", 2.0, input_tokens=100, output_tokens=50))
    store.save_result(_result("b", "completed", 4.0, input_tokens=200, output_tokens=50))
    store.save_result(_result("c", "failed", 9.0))  # failed runs excluded from latency
    stats = store.stats()
    assert stats["total_queries"] == 3
    assert stats["successful_queries"] == 2
    assert stats["failed_queries"] == 1
    assert stats["avg_latency"] == 3.0
    assert stats["queries_today"] == 3
    assert stats["total_tokens"] == 400
    assert stats["avg_tokens_per_query"] == round(400 / 3, 1)


def test_cost_is_none_without_pricing_not_zero():
    store = RuntimeStore()
    store.save_result(_result("a", input_tokens=10, estimated_cost=None))
    assert store.stats()["total_cost"] is None


def test_cost_is_summed_when_known():
    store = RuntimeStore()
    store.save_result(_result("a", estimated_cost=0.25))
    store.save_result(_result("b", estimated_cost=0.5))
    assert store.stats()["total_cost"] == 0.75
