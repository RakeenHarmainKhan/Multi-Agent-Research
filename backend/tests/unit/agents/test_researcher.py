import pytest

from app.agents import nodes


class FakeTool:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.queries = []

    def invoke(self, query):
        self.queries.append(query)
        if self.error:
            raise self.error
        return self.result


def _hit(title, url, content="body"):
    return {"title": title, "url": url, "content": content, "score": 0.9}


@pytest.fixture
def tools(monkeypatch):
    web = FakeTool(result=[])
    academic = FakeTool(result="")
    monkeypatch.setattr(nodes, "web_search", web)
    monkeypatch.setattr(nodes, "academic_search", academic)
    return web, academic


def _state(plan, **extra):
    return {"query": "q", "plan": plan, "requires_web": True, "requires_academic": False, **extra}


class TestResearcher:
    def test_collects_web_results(self, tools):
        web, _ = tools
        web.result = [_hit("A", "http://a")]
        out = nodes.researcher_node(_state([{"query": "x", "type": "web"}]))
        assert out["current_step"] == "writing"
        assert out["web_results"][0]["title"] == "A"
        assert out["web_results"][0]["source_query"] == "x"

    def test_accepts_tavily_dict_shape(self, tools):
        """TavilySearch returns {"results": [...]}, not a bare list."""
        web, _ = tools
        web.result = {"results": [_hit("A", "http://a")], "answer": "..."}
        out = nodes.researcher_node(_state([{"query": "x", "type": "web"}]))
        assert [r["url"] for r in out["web_results"]] == ["http://a"]

    def test_dedupes_urls_across_steps_ignoring_case_and_trailing_slash(self, tools):
        web, _ = tools
        web.result = [_hit("A", "https://Site.com/Post/"), _hit("A dup", "https://site.com/post")]
        out = nodes.researcher_node(
            _state([{"query": "x", "type": "web"}, {"query": "y", "type": "web"}])
        )
        assert len(out["web_results"]) == 1

    def test_limits_to_three_results_per_step_and_truncates_content(self, tools):
        web, _ = tools
        web.result = [_hit(f"T{i}", f"http://{i}", "z" * 2000) for i in range(6)]
        out = nodes.researcher_node(_state([{"query": "x", "type": "web"}]))
        assert len(out["web_results"]) == 3
        assert len(out["web_results"][0]["content"]) == 800

    def test_empty_plan_falls_back_to_query(self, tools):
        web, _ = tools
        nodes.researcher_node(_state([]))
        assert web.queries == ["q"]

    def test_academic_steps_need_flag(self, tools):
        _, academic = tools
        academic.result = "Title: P1\n\nTitle: P2\n\nTitle: P3"
        plan = [{"query": "x", "type": "academic"}]
        off = nodes.researcher_node(_state(plan, requires_academic=False))
        assert off["academic_results"] == []
        on = nodes.researcher_node(_state(plan, requires_academic=True))
        assert len(on["academic_results"]) == 2  # max 2 papers per step

    def test_web_steps_skipped_when_web_not_required(self, tools):
        web, _ = tools
        nodes.researcher_node(_state([{"query": "x", "type": "web"}], requires_web=False))
        assert web.queries == []

    def test_failing_step_does_not_abort_the_run(self, tools):
        web, _ = tools
        web.error = RuntimeError("tavily down")
        out = nodes.researcher_node(_state([{"query": "x", "type": "web"}]))
        assert out["web_results"] == []
        assert out["current_step"] == "writing"
