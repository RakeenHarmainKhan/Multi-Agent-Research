"""API routes for research execution and lightweight admin data."""
from datetime import datetime, timezone
import logging
import time
import uuid
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from app.agents.workflow import research_graph
from app.api.models import FeedbackRequest, HealthCheck, ResearchRequest, ResearchResponse, SystemStats
from app.core.config import settings
from app.core.runtime_store import runtime_store
from app.core.usage import TokenUsageHandler, estimate_cost

logger = logging.getLogger(__name__)
router = APIRouter()
_PROCESS_START = time.time()

_FRESH_RUN_STATE: Dict[str, Any] = {
    "intent": "",
    "requires_academic": False,
    "requires_web": True,
    "plan": [],
    "web_results": [],
    "academic_results": [],
    "all_sources": [],
    "final_report": "",
    "citations": [],
    "error": "",
    "current_step": "",
}


def _to_response(record: Dict[str, Any]) -> ResearchResponse:
    """Build the API response from a stored run record."""
    return ResearchResponse(
        thread_id=record["thread_id"],
        status=record["status"],
        report=record.get("final_report"),
        citations=record.get("citations", []),
        sources=record.get("sources", []),
        execution_time=record.get("execution_time"),
        input_tokens=record.get("input_tokens", 0),
        output_tokens=record.get("output_tokens", 0),
        estimated_cost=record.get("estimated_cost"),
    )


@router.post("/research", response_model=ResearchResponse)
async def execute_research(request: ResearchRequest):
    """Execute the research workflow."""
    thread_id = request.thread_id or str(uuid.uuid4())
    run_id = str(uuid.uuid4())
    start_time = time.time()
    history_messages = [message.model_dump() for message in request.messages]
    usage_handler = TokenUsageHandler()

    try:
        config = {"configurable": {"thread_id": thread_id}, "callbacks": [usage_handler]}
        inputs = {
            # Per-run outputs must be reset: the checkpointer keeps state per thread_id and
            # only overwrites keys we send, so a chat answer would otherwise inherit the
            # previous research answer's sources and citations.
            **_FRESH_RUN_STATE,
            "messages": history_messages,
            "query": request.query,
            "thread_id": thread_id,
            "mode": request.mode,
            "user_feedback": request.user_feedback or "",
            "revision_count": 0,
            "approved": not settings.ENABLE_HUMAN_REVIEW,
            "start_time": start_time,
        }

        logger.info("Starting research for thread %s", thread_id)
        final_state = await research_graph.ainvoke(inputs, config=config)
        execution_time = time.time() - start_time
        usage = usage_handler.summary()

        awaiting_review = (
            settings.ENABLE_HUMAN_REVIEW
            and final_state.get("current_step") == "review"
            and not final_state.get("approved", False)
        )
        record = {
            "run_id": run_id,
            "thread_id": thread_id,
            "query": request.query,
            "status": "awaiting_review" if awaiting_review else "completed",
            "intent": final_state.get("intent", "UNKNOWN"),
            "final_report": final_state.get("final_report", ""),
            "sources": final_state.get("all_sources", []),
            "citations": final_state.get("citations", []),
            "execution_time": execution_time,
            "created_at": datetime.now(timezone.utc),
            **usage,
        }
        runtime_store.save_result(record)
        logger.info(
            "Run %s finished in %.1fs, tokens in/out %s/%s",
            run_id, execution_time, usage["input_tokens"], usage["output_tokens"],
        )
        return _to_response(record)
    except Exception as exc:
        logger.error("Research failed: %s", exc, exc_info=True)
        runtime_store.save_result(
            {
                "run_id": run_id,
                "thread_id": thread_id,
                "query": request.query,
                "status": "failed",
                "intent": "UNKNOWN",
                "final_report": "",
                "sources": [],
                "citations": [],
                "execution_time": time.time() - start_time,
                "created_at": datetime.now(timezone.utc),
                **usage_handler.summary(),
            }
        )
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/research/{thread_id}", response_model=ResearchResponse)
def get_research_result(thread_id: str):
    """Retrieve the latest research result for a thread."""
    record = runtime_store.get_result(thread_id)
    if not record:
        raise HTTPException(status_code=404, detail="Research not found")
    return _to_response(record)


@router.get("/runs/{run_id}", response_model=ResearchResponse)
def get_run(run_id: str):
    """Retrieve one specific question's result (a thread can contain many runs)."""
    record = runtime_store.get_run(run_id)
    if not record:
        raise HTTPException(status_code=404, detail="Run not found")
    return _to_response(record)


@router.post("/research/{thread_id}/feedback", response_model=ResearchResponse)
async def submit_feedback(thread_id: str, request: FeedbackRequest):
    """Resume workflow with human feedback when review mode is enabled."""
    if not settings.ENABLE_HUMAN_REVIEW:
        raise HTTPException(status_code=400, detail="Human review mode is disabled")

    existing = runtime_store.get_result(thread_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Research not found")

    usage_handler = TokenUsageHandler()
    config = {"configurable": {"thread_id": thread_id}, "callbacks": [usage_handler]}
    research_graph.update_state(
        config,
        {"user_feedback": request.feedback, "approved": request.approved},
    )
    final_state = await research_graph.ainvoke(None, config=config)

    # The revision belongs to the same run: add its tokens/cost to the original run's totals.
    usage = usage_handler.summary()
    usage["input_tokens"] += existing.get("input_tokens", 0)
    usage["output_tokens"] += existing.get("output_tokens", 0)
    usage["estimated_cost"] = estimate_cost(usage["input_tokens"], usage["output_tokens"])

    record = {
        **existing,
        "status": "completed" if final_state.get("approved") else "awaiting_review",
        "intent": final_state.get("intent", existing.get("intent", "UNKNOWN")),
        "final_report": final_state.get("final_report", ""),
        "sources": final_state.get("all_sources", []),
        "citations": final_state.get("citations", []),
        **usage,
    }
    runtime_store.save_result(record)
    return _to_response(record)


@router.get("/admin/stats", response_model=SystemStats)
def get_system_stats():
    """System statistics for the current backend process."""
    return SystemStats(**runtime_store.stats())


@router.get("/admin/history")
def get_query_history(limit: int = 20):
    """Recent research queries from the in-memory store."""
    queries = runtime_store.list_history(limit=limit)
    return [
        {
            "id": index + 1,
            "run_id": item["run_id"],
            "thread_id": item["thread_id"],
            "query": item["query"][:100] + "..." if len(item["query"]) > 100 else item["query"],
            "status": item["status"],
            "intent": item.get("intent"),
            "execution_time": item.get("execution_time"),
            "total_tokens": item.get("input_tokens", 0) + item.get("output_tokens", 0),
            "estimated_cost": item.get("estimated_cost"),
            "created_at": item.get("created_at").isoformat() if item.get("created_at") else None,
        }
        for index, item in enumerate(queries)
    ]


@router.get("/health", response_model=HealthCheck)
def health_check():
    """Health check endpoint."""
    return HealthCheck(
        status="healthy",
        version=settings.VERSION,
        uptime=round(time.time() - _PROCESS_START, 1),
        database="in-memory",
        llm_service="not_configured" if settings.missing_llm_settings() else "configured",
        timestamp=datetime.now(timezone.utc),
    )