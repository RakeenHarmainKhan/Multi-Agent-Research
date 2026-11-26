"""LangGraph workflow orchestration."""
import logging
import time
from functools import wraps

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from app.observability.tracing import trace_span
from app.agents.state import ResearchState
from app.agents.nodes import (
    input_guard_node,
    intent_classifier_node,
    planner_node,
    researcher_node,
    writer_node,
    simple_chat_node,
    llm_retry_metrics,
)

logger = logging.getLogger(__name__)


def _timed_node(name, node):
    @wraps(node)
    def run(state):
        start = time.perf_counter()
        retry_metrics = {"retry_count": 0, "retry_wait_seconds": 0.0}
        token = llm_retry_metrics.set(retry_metrics)
        try:
            with trace_span(name) as metadata:
                try:
                    result = node(state)
                finally:
                    metadata.update(retry_metrics)
                if name == "intent_classifier":
                    metadata.setdefault("status", "blocked" if result.get("guardrails", {}).get("blocked") else "accepted")
        finally:
            llm_retry_metrics.reset(token)
            elapsed = time.perf_counter() - start
            logger.info("Workflow stage %s took %.3fs", name, elapsed)
        timings = {} if name == "input_guard" else dict(state.get("stage_timings") or {})
        retries = {} if name == "input_guard" else dict(state.get("stage_retries") or {})
        timings[name] = round(elapsed, 3)
        retries[name] = {**retry_metrics, "retry_wait_seconds": round(retry_metrics["retry_wait_seconds"], 3)}
        return {**result, "stage_timings": timings, "stage_retries": retries}
    return run


def create_research_workflow():
    """Build and compile the research agent workflow graph."""
    workflow = StateGraph(ResearchState)

    workflow.add_node("input_guard", _timed_node("input_guard", input_guard_node))
    workflow.add_node("intent_classifier", _timed_node("intent_classifier", intent_classifier_node))
    workflow.add_node("simple_chat", _timed_node("simple_chat", simple_chat_node))
    workflow.add_node("planner", _timed_node("planner", planner_node))
    workflow.add_node("researcher", _timed_node("researcher", researcher_node))
    workflow.add_node("writer", _timed_node("writer", writer_node))

    def route_after_guard(state: ResearchState) -> str:
        """A blocked query ends the run with a refusal."""
        return "end" if state.get("guardrails", {}).get("blocked") else "intent_classifier"

    def route_after_intent(state: ResearchState) -> str:
        """CHAT answers directly, LOOKUP already has its one-step plan, RESEARCH plans first."""
        current_step = state.get("current_step")
        if current_step == "chat":
            return "simple_chat"
        elif current_step == "completed":
            return "end"
        elif current_step == "research":
            return "researcher"
        return "planner"

    workflow.set_entry_point("input_guard")

    workflow.add_conditional_edges(
        "input_guard",
        route_after_guard,
        {"intent_classifier": "intent_classifier", "end": END},
    )

    workflow.add_conditional_edges(
        "intent_classifier",
        route_after_intent,
        {
            "simple_chat": "simple_chat",
            "planner": "planner",
            "researcher": "researcher",
            "end": END
        }
    )

    workflow.add_edge("simple_chat", END)

    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "writer")
    workflow.add_edge("writer", END)

    return workflow.compile(checkpointer=MemorySaver())


research_graph = create_research_workflow()
