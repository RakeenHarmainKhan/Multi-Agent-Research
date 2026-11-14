import pytest
from fastapi.testclient import TestClient

from app.api import routes
from app.core.runtime_store import runtime_store
from main import app


class FakeGraph:
    """Replaces research_graph so API tests never hit the LLM."""

    def __init__(self, state=None, error=None):
        self.state = state or {
            "intent": "RESEARCH",
            "final_report": "Report body [1]",
            "citations": [{"id": 1, "type": "web", "title": "T", "url": "http://t", "snippet": "s"}],
            "all_sources": [{"id": 1, "type": "web", "title": "T", "url": "http://t", "content": "c"}],
            "current_step": "completed",
            "approved": True,
        }
        self.error = error
        self.inputs = None

    async def ainvoke(self, inputs, config=None):
        self.inputs = inputs
        if self.error:
            raise self.error
        return self.state


@pytest.fixture
def client():
    runtime_store.clear()
    return TestClient(app)


@pytest.fixture
def graph(monkeypatch):
    fake = FakeGraph()
    monkeypatch.setattr(routes, "research_graph", fake)
    return fake


class TestMeta:
    def test_root(self, client):
        body = client.get("/").json()
        assert body["docs"] == "/docs"

    def test_health(self, client):
        body = client.get("/api/v1/health").json()
        assert body["status"] == "healthy"
        assert body["database"] == "in-memory"

    def test_health_uptime_is_seconds_since_start_not_epoch(self, client):
        """Regression: uptime used to return time.time() (~1.8e9)."""
        uptime = client.get("/api/v1/health").json()["uptime"]
        assert 0 <= uptime < 24 * 3600

    def test_health_reports_llm_not_configured_when_settings_missing(self, client, monkeypatch):
        monkeypatch.setattr(routes.settings, "AZURE_OPENAI_ENDPOINT", "")
        assert client.get("/api/v1/health").json()["llm_service"] == "not_configured"


class TestResearchEndpoint:
    def test_success_returns_report_citations_and_sources(self, client, graph):
        r = client.post("/api/v1/research", json={"query": "lora"})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "completed"
        assert body["report"] == "Report body [1]"
        assert body["citations"][0]["id"] == 1
        assert body["sources"][0]["id"] == 1
        assert body["thread_id"]

    def test_mode_defaults_to_auto(self, client, graph):
        client.post("/api/v1/research", json={"query": "lora"})
        assert graph.inputs["mode"] == "auto"

    def test_client_cannot_override_model_or_key(self, client, graph):
        """Old clients may still send these; they are ignored, never forwarded to the graph."""
        r = client.post("/api/v1/research", json={"query": "q", "model": "x", "api_key": "sk-123"})
        assert r.status_code == 200
        assert "api_key" not in graph.inputs and "selected_model" not in graph.inputs

    def test_research_mode_is_forwarded_to_graph(self, client, graph):
        client.post("/api/v1/research", json={"query": "lora", "mode": "research"})
        assert graph.inputs["mode"] == "research"

    def test_history_messages_are_forwarded(self, client, graph):
        client.post(
            "/api/v1/research",
            json={"query": "and cost?", "messages": [{"role": "user", "content": "about lora"}]},
        )
        assert graph.inputs["messages"] == [{"role": "user", "content": "about lora"}]

    def test_thread_id_is_reused_when_supplied(self, client, graph):
        r = client.post("/api/v1/research", json={"query": "q", "thread_id": "abc"})
        assert r.json()["thread_id"] == "abc"

    def test_follow_up_after_long_report_is_accepted(self, client, graph):
        """Regression: research reports exceed 2k chars and used to 422 the next question."""
        history = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "x" * 3000},
        ]
        r = client.post("/api/v1/research", json={"query": "and its cost?", "messages": history})
        assert r.status_code == 200

    @pytest.mark.parametrize(
        "payload",
        [
            {"query": ""},
            {},
            {"query": "q", "mode": "bogus"},
            {"query": "x" * 1501},
            {"query": "q", "messages": [{"role": "robot", "content": "hi"}]},
            {"query": "q", "messages": [{"role": "user", "content": "m"}] * 51},
        ],
    )
    def test_invalid_requests_are_rejected(self, client, graph, payload):
        assert client.post("/api/v1/research", json=payload).status_code == 422

    def test_graph_failure_returns_500_and_records_failed_run(self, client, monkeypatch):
        monkeypatch.setattr(routes, "research_graph", FakeGraph(error=RuntimeError("boom")))
        r = client.post("/api/v1/research", json={"query": "q", "thread_id": "bad"})
        assert r.status_code == 500
        assert runtime_store.get_result("bad")["status"] == "failed"


class TestResultRetrieval:
    def test_get_saved_result(self, client, graph):
        tid = client.post("/api/v1/research", json={"query": "q"}).json()["thread_id"]
        r = client.get(f"/api/v1/research/{tid}")
        assert r.status_code == 200
        assert r.json()["report"] == "Report body [1]"

    def test_unknown_thread_is_404(self, client):
        assert client.get("/api/v1/research/nope").status_code == 404


class TestFeedback:
    def test_disabled_when_human_review_is_off(self, client):
        r = client.post("/api/v1/research/t1/feedback", json={"feedback": "x", "approved": False})
        assert r.status_code == 400


class TestAdmin:
    def test_stats_and_history_reflect_runs(self, client, graph):
        client.post("/api/v1/research", json={"query": "first question"})
        client.post("/api/v1/research", json={"query": "second question"})
        stats = client.get("/api/v1/admin/stats").json()
        assert stats["total_queries"] == 2
        assert stats["successful_queries"] == 2
        history = client.get("/api/v1/admin/history").json()
        assert [h["query"] for h in history] == ["second question", "first question"]

    def test_history_truncates_long_queries(self, client, graph):
        client.post("/api/v1/research", json={"query": "q" * 300})
        assert client.get("/api/v1/admin/history").json()[0]["query"].endswith("...")

    def test_every_question_in_a_conversation_is_counted(self, client, graph):
        """Regression: runs were keyed by thread_id, so a whole conversation counted as one query."""
        client.post("/api/v1/research", json={"query": "first", "thread_id": "chat-1"})
        client.post("/api/v1/research", json={"query": "follow-up", "thread_id": "chat-1"})
        assert client.get("/api/v1/admin/stats").json()["total_queries"] == 2
        history = client.get("/api/v1/admin/history").json()
        assert [h["query"] for h in history] == ["follow-up", "first"]
        assert client.get("/api/v1/research/chat-1").json()["status"] == "completed"

    def test_each_history_entry_opens_its_own_report(self, client, monkeypatch):
        """Two questions in one thread must not both open the latest report."""
        fake = FakeGraph()
        monkeypatch.setattr(routes, "research_graph", fake)
        fake.state = {**fake.state, "final_report": "first report"}
        client.post("/api/v1/research", json={"query": "first", "thread_id": "chat-1"})
        fake.state = {**fake.state, "final_report": "second report"}
        client.post("/api/v1/research", json={"query": "second", "thread_id": "chat-1"})

        history = client.get("/api/v1/admin/history").json()
        reports = {h["query"]: client.get(f"/api/v1/runs/{h['run_id']}").json()["report"] for h in history}
        assert reports == {"first": "first report", "second": "second report"}

    def test_unknown_run_is_404(self, client):
        assert client.get("/api/v1/runs/nope").status_code == 404


class TestUsageAccounting:
    """Runs the REAL graph with a fake chat model that reports token usage."""

    @pytest.fixture
    def real_graph(self, monkeypatch):
        import json

        from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
        from langchain_core.messages import AIMessage

        from app.agents import nodes

        def reply(text, i, o):
            return AIMessage(
                content=text,
                usage_metadata={"input_tokens": i, "output_tokens": o, "total_tokens": i + o},
            )

        replies = iter(
            [
                reply(json.dumps({"steps": [{"query": "w", "type": "web"}]}), 100, 20),  # planner
                reply("Fact [1].", 300, 80),  # writer
            ]
        )
        monkeypatch.setattr(nodes, "_build_llm", lambda: GenericFakeChatModel(messages=replies))

        class Web:
            def invoke(self, q):
                return [{"title": "A", "url": "http://a", "content": "c"}]

        monkeypatch.setattr(nodes, "web_search", Web())

    def test_tokens_from_every_llm_call_are_summed(self, client, real_graph):
        body = client.post("/api/v1/research", json={"query": "q", "mode": "research"}).json()
        assert body["input_tokens"] == 400
        assert body["output_tokens"] == 100

    def test_cost_is_null_without_pricing(self, client, real_graph):
        body = client.post("/api/v1/research", json={"query": "q", "mode": "research"}).json()
        assert body["estimated_cost"] is None
        assert client.get("/api/v1/admin/stats").json()["total_cost"] is None

    def test_cost_uses_configured_pricing(self, client, real_graph, monkeypatch):
        monkeypatch.setattr(routes.settings, "LLM_INPUT_COST_PER_1M_TOKENS", 2.0)
        monkeypatch.setattr(routes.settings, "LLM_OUTPUT_COST_PER_1M_TOKENS", 10.0)
        body = client.post("/api/v1/research", json={"query": "q", "mode": "research"}).json()
        assert body["estimated_cost"] == pytest.approx(400 / 1e6 * 2 + 100 / 1e6 * 10)

    def test_stats_report_real_tokens(self, client, real_graph):
        client.post("/api/v1/research", json={"query": "q", "mode": "research"})
        stats = client.get("/api/v1/admin/stats").json()
        assert stats["total_tokens"] == 500
        assert "active_users" not in stats


class TestRunIsolationWithinAThread:
    """Real graph + checkpointer: consecutive questions in one thread must not share outputs."""

    @pytest.fixture
    def real_graph(self, monkeypatch, llm_response):
        import json

        from app.agents import nodes

        def fake_llm(state, prompt_messages):
            system = str(prompt_messages[0].content)
            if "Break down the research query" in system:
                return llm_response(json.dumps({"steps": [{"query": "w", "type": "web"}]}))
            if "research summary" in system:
                return llm_response("Fact [1].")
            if "ONLY respond to queries related to Artificial Intelligence" in system:
                return llm_response(
                    json.dumps(
                        {"intent": "CHAT", "requires_academic": False, "requires_web": False, "reasoning": "r"}
                    )
                )
            return llm_response("Plain chat answer.")

        class Web:
            def invoke(self, q):
                return [{"title": "A", "url": "http://a", "content": "c"}]

        monkeypatch.setattr(nodes, "_invoke_with_llm", fake_llm)
        monkeypatch.setattr(nodes, "web_search", Web())

    def test_chat_after_research_has_no_stale_sources(self, client, real_graph):
        """Regression: the chat answer inherited the previous research answer's citations."""
        first = client.post(
            "/api/v1/research", json={"query": "q1", "mode": "research", "thread_id": "iso-1"}
        ).json()
        assert first["citations"]

        second = client.post(
            "/api/v1/research", json={"query": "hi", "mode": "auto", "thread_id": "iso-1"}
        ).json()
        assert second["report"] == "Plain chat answer."
        assert second["citations"] == []
        assert second["sources"] == []
