"""
Shared pytest setup.

Tests never call a real LLM or search API (both are replaced with fakes), so no
model, endpoint or key is configured here. Environment variables are set before
any `app.*` import so a developer's local backend/.env cannot change behaviour.
"""
import os

# Deterministic behaviour regardless of the developer's local .env
os.environ["TAVILY_API_KEY"] = ""
os.environ["ENABLE_HUMAN_REVIEW"] = "false"
os.environ["BACKEND_CORS_ORIGINS"] = "http://localhost:5173,https://*.vercel.app"

from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402


class FakeLLMResponse(SimpleNamespace):
    """Minimal stand-in for a LangChain AIMessage."""

    def __init__(self, content: str):
        super().__init__(content=content)


@pytest.fixture
def llm_response():
    return FakeLLMResponse


@pytest.fixture
def prompt_text():
    """Flatten formatted prompt messages into one string for assertions."""

    def _flatten(prompt_messages) -> str:
        return "\n".join(str(m.content) for m in prompt_messages)

    return _flatten
