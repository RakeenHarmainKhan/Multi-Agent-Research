"""
End-to-end graph tests: the real LangGraph runs, with a scripted LLM and fake search tools.
"""
import json
import uuid

import pytest

from app.agents import nodes
from app.agents.workflow import research_graph


class Script:
    """Routes each LLM call by recognising which node's prompt it is."""

    def __init__(self):
        self.calls = []
        self.intent = "RESEARCH"
        self.chat_reply = "Plain chat answer."
        self.report = "Finding one [1]. Finding two [2]. Bogus [9]."

    def __call__(self, state, prompt_messages, llm_response):
        system = str(prompt_messages[0].content)
        if "ONLY respond to queries related to Artificial Intelligence" in system:
            self.calls.append("classifier")
            if self.intent == "REJECTED":
                return llm_response("REJECTED")
            return llm_response(
                json.dumps(
                    {
                        "intent": self.intent,
                        "requires_academic": True,
                        "requires_web": True,
                        "reasoning": "r",
                    }
                )
            )
        if "Break down the research query" in system:
            self.calls.append("planner")
            return llm_response(
                json.dumps(
                    {
                        "steps": [
                            {"query": "web q", "type": "web"},
                            {"query": "paper q", "type": "academic"},
                        ]
                    }
                )
            )
        if "research summary" in system:
            self.calls.append("writer")
            return llm_response(self.report)
        self.calls.append("chat")
        self.last_chat_system = system
        return llm_response(self.chat_reply)


class FakeTool:
    def __init__(self, result):
        self.result = result
        self.queries = []

    def invoke(self, query):
        self.queries.append(query)
        return self.result


@pytest.fixture
def env(monkeypatch, llm_response):
    script = Script()
    web = FakeTool(
        {
            "results": [
                {"title": "Alpha", "url": "http://alpha", "content": "a", "score": 1},
                {"title": "Beta", "url": "http://beta", "content": "b", "score": 1},
            ]
        }
    )
    academic = FakeTool("Published: 2025\nTitle: Paper X\nAuthors: Z")
    monkeypatch.setattr(nodes, "_invoke_with_llm", lambda s, m: script(s, m, llm_response))
    monkeypatch.setattr(nodes, "web_search", web)
    monkeypatch.setattr(nodes, "academic_search", academic)
    return script, web, academic


def run(query="q", mode="auto"):
    state = {
        "query": query,
        "mode": mode,
        "messages": [],
        "approved": True,  # no human review in tests
        "revision_count": 0,
    }
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    return research_graph.invoke(state, config=config)


class TestAutoMode:
    def test_chat_intent_skips_search(self, env):
        script, web, academic = env
        script.intent = "CHAT"
        out = run("What is a neural network?")
        assert script.calls == ["classifier", "chat"]
        assert web.queries == [] and academic.queries == []
        assert out["final_report"] == "Plain chat answer."
        assert not out.get("all_sources")

    def test_research_intent_runs_full_pipeline(self, env):
        script, web, academic = env
        out = run("Latest LLM architectures")
        assert script.calls == ["classifier", "planner", "writer"]
        assert web.queries == ["web q"] and academic.queries == ["paper q"]
        assert out["current_step"] == "completed"

    def test_citations_resolve_to_retrieved_sources(self, env):
        out = run("Latest LLM architectures")
        assert "[9]" not in out["final_report"]
        assert [c["title"] for c in out["citations"]] == ["Alpha", "Beta"]
        assert [s["id"] for s in out["all_sources"]] == [1, 2, 3]

    def test_rejected_query_gets_brief_out_of_scope_answer(self, env):
        script, web, _ = env
        script.intent = "REJECTED"
        out = run("How to bake pizza?")
        assert script.calls == ["classifier", "chat"]
        assert "outside your research scope" in script.last_chat_system
        assert web.queries == []
        assert out["intent"] == "REJECTED"


class TestResearchMode:
    def test_skips_classifier_and_always_searches(self, env):
        script, web, _ = env
        script.intent = "CHAT"  # would have been chat in auto mode
        out = run("What is a neural network?", mode="research")
        assert "classifier" not in script.calls
        assert script.calls == ["planner", "writer"]
        assert web.queries == ["web q"]
        assert out["intent"] == "RESEARCH"
        assert out["citations"]

    def test_same_query_in_auto_mode_would_not_search(self, env):
        script, web, _ = env
        script.intent = "CHAT"
        run("What is a neural network?", mode="auto")
        assert web.queries == []
