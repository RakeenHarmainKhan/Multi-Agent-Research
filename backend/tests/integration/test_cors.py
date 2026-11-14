import re

import pytest
from fastapi.testclient import TestClient

from app.core.config import cors_options
from main import app


class TestCorsOptions:
    def test_exact_origins_stay_exact(self):
        opts = cors_options(["http://localhost:5173"])
        assert opts == {"allow_origins": ["http://localhost:5173"], "allow_origin_regex": None}

    def test_wildcard_becomes_anchored_regex(self):
        regex = cors_options(["https://*.vercel.app"])["allow_origin_regex"]
        assert re.fullmatch(regex, "https://my-app.vercel.app")
        assert not re.fullmatch(regex, "https://vercel.app")
        assert not re.fullmatch(regex, "https://a.b.vercel.app")  # one label only
        assert not re.fullmatch(regex, "https://evil.com/?.vercel.app")

    def test_dots_are_escaped(self):
        regex = cors_options(["https://*.vercel.app"])["allow_origin_regex"]
        assert not re.fullmatch(regex, "https://my-appXvercelXapp")

    def test_bare_star_allows_all(self):
        assert cors_options(["*"]) == {"allow_origins": ["*"], "allow_origin_regex": None}

    def test_mixed(self):
        opts = cors_options(["http://localhost:5173", "https://*.onrender.com"])
        assert opts["allow_origins"] == ["http://localhost:5173"]
        assert re.fullmatch(opts["allow_origin_regex"], "https://mara.onrender.com")


def _preflight(origin):
    return TestClient(app).options(
        "/api/v1/research",
        headers={"Origin": origin, "Access-Control-Request-Method": "POST"},
    )


@pytest.mark.parametrize("origin", ["http://localhost:5173", "https://mara-frontend.vercel.app"])
def test_preflight_allows_configured_origins(origin):
    """Regression: the *.vercel.app entry used to match nothing."""
    response = _preflight(origin)
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin


@pytest.mark.parametrize("origin", ["https://evil.com", "https://vercel.app.evil.com"])
def test_preflight_rejects_other_origins(origin):
    response = _preflight(origin)
    assert "access-control-allow-origin" not in response.headers
