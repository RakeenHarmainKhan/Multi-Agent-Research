"""In-memory runtime store for stateless-friendly deployments."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Deque, Dict, List

from app.core.config import settings


@dataclass
class RuntimeStore:
    """
    Stores recent research runs in process memory.

    One record per run (one question). A conversation (thread_id) has many runs;
    `get_result(thread_id)` returns the latest one.
    """

    _lock: Lock = field(default_factory=Lock)
    _runs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    _latest_run_by_thread: Dict[str, str] = field(default_factory=dict)
    _history: Deque[str] = field(default_factory=lambda: deque(maxlen=settings.HISTORY_LIMIT))

    def save_result(self, result: Dict[str, Any]) -> None:
        """Insert a run, or update it if `run_id` already exists (e.g. after human feedback)."""
        with self._lock:
            run_id = result["run_id"]
            payload = {**result, "updated_at": datetime.now(timezone.utc)}
            if run_id not in self._runs:
                if len(self._history) == self._history.maxlen:
                    evicted = self._history[-1]
                    evicted_thread = self._runs.pop(evicted, {}).get("thread_id")
                    if self._latest_run_by_thread.get(evicted_thread) == evicted:
                        del self._latest_run_by_thread[evicted_thread]
                self._history.appendleft(run_id)
            self._runs[run_id] = payload
            self._latest_run_by_thread[result["thread_id"]] = run_id

    def get_result(self, thread_id: str) -> Dict[str, Any] | None:
        """Latest run for a conversation."""
        with self._lock:
            run_id = self._latest_run_by_thread.get(thread_id)
            result = self._runs.get(run_id) if run_id else None
            return dict(result) if result else None

    def get_run(self, run_id: str) -> Dict[str, Any] | None:
        with self._lock:
            result = self._runs.get(run_id)
            return dict(result) if result else None

    def list_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(self._runs[run_id]) for run_id in list(self._history)[:limit] if run_id in self._runs]

    def clear(self) -> None:
        with self._lock:
            self._runs.clear()
            self._latest_run_by_thread.clear()
            self._history.clear()

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            results = list(self._runs.values())
            total = len(results)
            successful = sum(1 for item in results if item.get("status") == "completed")
            failed = sum(1 for item in results if item.get("status") == "failed")
            completed_times = [
                item.get("execution_time")
                for item in results
                if item.get("execution_time") is not None and item.get("status") == "completed"
            ]
            today = datetime.now(timezone.utc).date()
            queries_today = sum(
                1
                for item in results
                if item.get("created_at") and item["created_at"].date() == today
            )
            token_counts = [
                item.get("input_tokens", 0) + item.get("output_tokens", 0) for item in results
            ]
            costs = [item["estimated_cost"] for item in results if item.get("estimated_cost") is not None]

            return {
                "total_queries": total,
                "successful_queries": successful,
                "failed_queries": failed,
                "avg_latency": round(sum(completed_times) / len(completed_times), 2) if completed_times else 0.0,
                "total_tokens": sum(token_counts),
                "avg_tokens_per_query": round(sum(token_counts) / total, 1) if total else 0.0,
                # None (not 0) when pricing isn't configured, so the UI can't show a fake $0.00
                "total_cost": round(sum(costs), 6) if costs else None,
                "queries_today": queries_today,
            }


runtime_store = RuntimeStore()
