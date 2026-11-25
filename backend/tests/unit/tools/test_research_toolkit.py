import pytest

from app.tools import research
from app.tools.research import ResearchToolkit


class _FakeTavily:
    """Records constructor kwargs; mimics TavilySearch's signature requirement."""

    calls = []

    def __init__(self, **kwargs):
        if "tavily_api_key" not in kwargs:
            raise ValueError("Did not find tavily_api_key")
        _FakeTavily.calls.append(kwargs)


class TestWebSearchTool:
    def test_without_key_returns_no_fake_sources(self):
        toolkit = ResearchToolkit()
        toolkit.tavily_available = False
        with pytest.raises(research.WebSearchUnavailableError):
            toolkit.get_web_search_tool().invoke("llm news")

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
        assert _FakeTavily.calls[0]["include_answer"] is False

    def test_constructor_failure_falls_back_to_mock(self, monkeypatch):
        def boom(**kwargs):
            raise RuntimeError("bad key")

        monkeypatch.setattr(research, "TavilySearch", boom)
        monkeypatch.setattr(research.settings, "TAVILY_API_KEY", "tvly-123")
        tool = ResearchToolkit().get_web_search_tool()
        with pytest.raises(research.WebSearchUnavailableError):
            tool.invoke("q")


class TestAcademicSearch:
    def test_is_tavily_limited_to_scholarly_sites(self, monkeypatch):
        _FakeTavily.calls.clear()
        monkeypatch.setattr(research, "TavilySearch", _FakeTavily)
        monkeypatch.setattr(research.settings, "TAVILY_API_KEY", "tvly-123")
        ResearchToolkit().get_academic_search_tool()
        ResearchToolkit().get_web_search_tool()
        assert _FakeTavily.calls[0]["include_domains"] == research.ACADEMIC_DOMAINS
        assert _FakeTavily.calls[1]["include_domains"] is None

    @pytest.mark.parametrize("message, transient", [
        ("Page request resulted in HTTP 429", True),
        ("Page request resulted in HTTP 503", True),
        ("Page request resulted in HTTP 500", True),
        ("HTTP 401 unauthorized", False),
    ])
    def test_transient_error_detection(self, message, transient):
        assert research.is_transient_search_error(RuntimeError(message)) is transient


class TestExtractPassage:
    def test_requests_one_query_ranked_passage_and_closes_session(self, monkeypatch):
        calls = []

        class FakeSession:
            def close(self):
                calls.append("closed")

        class FakeClient:
            def __init__(self, api_key):
                self.session = FakeSession()
                calls.append(api_key)

            def extract(self, **kwargs):
                calls.append(kwargs)
                return {"results": [{"url": "https://example.com/paper", "raw_content": " Selected passage "}]}

        monkeypatch.setattr(research.settings, "TAVILY_API_KEY", "test-key")
        monkeypatch.setattr(research, "TavilyClient", FakeClient)
        assert research.extract_passage("https://example.com/paper", "winner names") == "Selected passage"
        assert calls == [
            "test-key",
            {"urls": ["https://example.com/paper"], "query": "winner names", "chunks_per_source": 1, "timeout": 10},
            "closed",
        ]

    def test_rejects_mismatched_page_and_empty_results(self, monkeypatch):
        class FakeClient:
            session = type("Session", (), {"close": lambda self: None})()
            results = []

            def __init__(self, api_key):
                pass

            def extract(self, **kwargs):
                return {"results": self.results}

        monkeypatch.setattr(research.settings, "TAVILY_API_KEY", "test-key")
        monkeypatch.setattr(research, "TavilyClient", FakeClient)
        assert research.extract_passage("https://example.com/a", "q") is None
        FakeClient.results = [{"url": "https://other.example/b", "raw_content": "facts"}]
        with pytest.raises(ValueError, match="URL"):
            research.extract_passage("https://example.com/a", "q")

    @pytest.mark.parametrize("response", [None, [], {}, {"results": "bad"}, {"results": [], "failed_results": [{"error": "failed"}]}])
    def test_rejects_malformed_or_failed_response(self, monkeypatch, response):
        closed = []

        class FakeClient:
            session = type("Session", (), {"close": lambda self: closed.append(True)})()

            def __init__(self, api_key):
                pass

            def extract(self, **kwargs):
                return response

        monkeypatch.setattr(research.settings, "TAVILY_API_KEY", "test-key")
        monkeypatch.setattr(research, "TavilyClient", FakeClient)
        with pytest.raises(ValueError):
            research.extract_passage("https://example.com/a", "q")
        assert closed == [True]
