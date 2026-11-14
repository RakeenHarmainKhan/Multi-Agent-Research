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
    runtime_store._results.clear()
    runtime_store._history.clear()
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
