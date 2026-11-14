from app.core.config import Settings


def _settings(**env):
    """Build Settings from the environment only, ignoring backend/.env."""
    return Settings(_env_file=None, **env)


def test_cors_comma_separated(monkeypatch):
    monkeypatch.setenv("BACKEND_CORS_ORIGINS", "http://a.com, http://b.com")
    assert _settings().BACKEND_CORS_ORIGINS == ["http://a.com", "http://b.com"]


def test_cors_json_list(monkeypatch):
    monkeypatch.setenv("BACKEND_CORS_ORIGINS", '["http://a.com"]')
    assert _settings().BACKEND_CORS_ORIGINS == ["http://a.com"]


def test_cors_empty_uses_defaults(monkeypatch):
    monkeypatch.setenv("BACKEND_CORS_ORIGINS", "")
    assert "http://localhost:5173" in _settings().BACKEND_CORS_ORIGINS


def test_human_review_flag_off_in_tests():
    assert _settings().ENABLE_HUMAN_REVIEW is False


def test_blank_llm_prices_mean_unset(monkeypatch):
    """Regression: `LLM_INPUT_COST_PER_1M_TOKENS=` (as in .env.example) crashed startup."""
    monkeypatch.setenv("LLM_INPUT_COST_PER_1M_TOKENS", "")
    monkeypatch.setenv("LLM_OUTPUT_COST_PER_1M_TOKENS", "  ")
    s = _settings()
    assert s.LLM_INPUT_COST_PER_1M_TOKENS is None
    assert s.LLM_OUTPUT_COST_PER_1M_TOKENS is None


def test_llm_prices_parse(monkeypatch):
    monkeypatch.setenv("LLM_INPUT_COST_PER_1M_TOKENS", "1.25")
    monkeypatch.setenv("LLM_OUTPUT_COST_PER_1M_TOKENS", "10")
    s = _settings()
    assert (s.LLM_INPUT_COST_PER_1M_TOKENS, s.LLM_OUTPUT_COST_PER_1M_TOKENS) == (1.25, 10.0)
