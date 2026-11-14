"""LangGraph workflow orchestration."""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from app.agents.state import ResearchState
from app.agents.nodes import (
    input_guard_node,
    intent_classifier_node,
    planner_node,
    researcher_node,
    writer_node,
    simple_chat_node,
)


def create_research_workflow():
    """Build and compile the research agent workflow graph."""
    workflow = StateGraph(ResearchState)

    workflow.add_node("input_guard", input_guard_node)
    workflow.add_node("intent_classifier", intent_classifier_node)
    workflow.add_node("simple_chat", simple_chat_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("writer", writer_node)

    def route_after_guard(state: ResearchState) -> str:
        """A blocked query ends the run with a refusal."""
        return "end" if state.get("guardrails", {}).get("blocked") else "intent_classifier"

    def route_after_intent(state: ResearchState) -> str:
        """Route based on intent classification."""
        current_step = state.get("current_step")
        if current_step == "chat":
            return "simple_chat"
        elif current_step == "completed":
            return "end"
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
            "end": END
        }
    )

    workflow.add_edge("simple_chat", END)

    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "writer")
    workflow.add_edge("writer", END)

    return workflow.compile(checkpointer=MemorySaver())


research_graph = create_research_workflow()
