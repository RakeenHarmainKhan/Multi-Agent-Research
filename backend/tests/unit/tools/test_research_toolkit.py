from app.tools import research
from app.tools.research import ResearchToolkit, academic_search


class _FakeTavily:
    """Records constructor kwargs; mimics TavilySearch's signature requirement."""

    calls = []

    def __init__(self, **kwargs):
        if "tavily_api_key" not in kwargs:
            raise ValueError("Did not find tavily_api_key")
        _FakeTavily.calls.append(kwargs)


class TestWebSearchTool:
    def test_without_key_uses_mock_fallback(self):
        toolkit = ResearchToolkit()
        toolkit.tavily_available = False
        results = toolkit.get_web_search_tool().invoke("llm news")
        assert results[0]["title"] == "Search API not configured"
        assert "llm news" in results[0]["content"]

    def test_with_key_passes_tavily_api_key(self, monkeypatch):
        """Regression: the installed TavilySearch needs `tavily_api_key`, not `api_key`."""
        _FakeTavily.calls.clear()
        monkeypatch.setattr(research, "TavilySearch", _FakeTavily)
        monkeypatch.setattr(research.settings, "TAVILY_API_KEY", "tvly-123")
        toolkit = ResearchToolkit()
        tool = toolkit.get_web_search_tool()
        assert isinstance(tool, _FakeTavily)
        assert _FakeTavily.calls[0]["tavily_api_key"] == "tvly-123"
        assert _FakeTavily.calls[0]["max_results"] == 3

    def test_constructor_failure_falls_back_to_mock(self, monkeypatch):
        def boom(**kwargs):
            raise RuntimeError("bad key")

        monkeypatch.setattr(research, "TavilySearch", boom)
        monkeypatch.setattr(research.settings, "TAVILY_API_KEY", "tvly-123")
        tool = ResearchToolkit().get_web_search_tool()
        assert tool.invoke("q")[0]["title"] == "Search API not configured"


class TestAcademicSearch:
    def test_returns_wrapper_output(self, monkeypatch):
        class FakeArxiv:
            def __init__(self, **kwargs):
                pass

            def run(self, query):
                return f"Published: 2025\nTitle: Result for {query}"

        monkeypatch.setattr(research, "ArxivAPIWrapper", FakeArxiv)
        assert "Result for lora" in academic_search.invoke({"query": "lora"})

    def test_failure_is_reported_not_raised(self, monkeypatch):
        class BrokenArxiv:
            def __init__(self, **kwargs):
                raise RuntimeError("arxiv down")

        monkeypatch.setattr(research, "ArxivAPIWrapper", BrokenArxiv)
        out = academic_search.invoke({"query": "lora"})
        assert out.startswith("Academic search unavailable")
