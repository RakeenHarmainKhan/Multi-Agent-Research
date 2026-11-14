"""Helper functions and retry behaviour in agents/nodes.py."""
import pytest

from app.agents import nodes


class TestConversationContext:
    def test_empty(self):
        assert nodes._conversation_context([]) == "No previous conversation context provided."

    def test_formats_roles_uppercase(self):
        out = nodes._conversation_context([{"role": "user", "content": "hi"}])
        assert out == "USER: hi"

    def test_limits_to_most_recent(self):
        msgs = [{"role": "user", "content": f"m{i}"} for i in range(10)]
        out = nodes._conversation_context(msgs, limit=2)
        assert out == "USER: m8\nUSER: m9"

    def test_truncates_long_content(self):
        out = nodes._conversation_context([{"role": "user", "content": "x" * 900}])
        assert len(out) == len("USER: ") + 500

    def test_skips_blank_messages(self):
        out = nodes._conversation_context([{"role": "user", "content": "   "}])
        assert out == "No previous conversation context provided."


class TestRateLimitDetection:
    @pytest.mark.parametrize("msg", ["Error 429", "Rate limit exceeded", "rate-limited", "HTTP 404"])
    def test_detected(self, msg):
        assert nodes._is_rate_limited_error(Exception(msg)) is True

    def test_other_errors_not_detected(self):
        assert nodes._is_rate_limited_error(Exception("invalid api key")) is False


class _FlakyLLM:
    def __init__(self, errors):
        self.errors = list(errors)
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return "ok"


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(nodes._invoke_with_retry.retry, "sleep", lambda _s: None)


class TestInvokeWithRetry:
    def test_retries_rate_limit_then_succeeds(self, no_sleep):
        llm = _FlakyLLM([Exception("429 rate limit")])
        assert nodes._invoke_with_retry(llm, []) == "ok"
        assert llm.calls == 2

    def test_gives_up_after_three_attempts_and_reraises(self, no_sleep):
        llm = _FlakyLLM([Exception("429")] * 5)
        with pytest.raises(Exception, match="429"):
            nodes._invoke_with_retry(llm, [])
        assert llm.calls == 3

    def test_non_rate_limit_error_is_not_retried(self, no_sleep):
        llm = _FlakyLLM([ValueError("boom")])
        with pytest.raises(ValueError):
            nodes._invoke_with_retry(llm, [])
        assert llm.calls == 1

    def test_successful_response_containing_404_text_is_not_retried(self, no_sleep):
        """Regression: the retry predicate must inspect the exception, not the result."""

        class OkLLM:
            calls = 0

            def invoke(self, messages):
                OkLLM.calls += 1
                return "Report mentioning HTTP 404 and 429"

        assert nodes._invoke_with_retry(OkLLM(), []) == "Report mentioning HTTP 404 and 429"
        assert OkLLM.calls == 1
