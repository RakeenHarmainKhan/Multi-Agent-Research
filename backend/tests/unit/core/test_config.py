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
