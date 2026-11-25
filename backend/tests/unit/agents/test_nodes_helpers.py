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


class TestRetryableErrorDetection:
    @pytest.mark.parametrize("msg", ["Error 429", "Rate limit exceeded", "rate-limited"])
    def test_detected(self, msg):
        assert nodes._is_retryable_llm_error(Exception(msg)) is True

    def test_dropped_connection_detected(self):
        assert nodes._is_retryable_llm_error(nodes.APIConnectionError(request=None)) is True

    @pytest.mark.parametrize("msg", ["invalid api key", "HTTP 404: DeploymentNotFound"])
    def test_other_errors_not_detected(self, msg):
        assert nodes._is_retryable_llm_error(Exception(msg)) is False


def test_llm_client_is_reused(monkeypatch):
    nodes._cached_llm.cache_clear()
    created = []
    monkeypatch.setattr(type(nodes.settings), "missing_llm_settings", lambda self: [])
    monkeypatch.setattr(nodes, "AzureChatOpenAI", lambda **kwargs: created.append(kwargs) or object())
    try:
        assert nodes._build_llm() is nodes._build_llm()
        assert len(created) == 1
    finally:
        nodes._cached_llm.cache_clear()


def test_llm_retries_are_attributed_to_the_current_stage(monkeypatch):
    from app.agents.workflow import _timed_node

    monkeypatch.setattr(nodes._invoke_with_retry.retry, "sleep", lambda _s: None)
    monkeypatch.setattr(nodes._invoke_with_retry.retry, "wait", lambda _state: 2)
    llm = _FlakyLLM([Exception("429")])
    node = _timed_node("writer", lambda state: {"answer": nodes._invoke_with_retry(llm, [])})
    result = node({})
    assert result["answer"] == "ok"
    assert result["stage_retries"]["writer"] == {"retry_count": 1, "retry_wait_seconds": 2}
    assert nodes.llm_retry_metrics.get() is None


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

    def test_retries_dropped_connection_then_succeeds(self, no_sleep):
        llm = _FlakyLLM([nodes.APIConnectionError(request=None)])
        assert nodes._invoke_with_retry(llm, []) == "ok"
        assert llm.calls == 2

    def test_gives_up_after_five_attempts_and_reraises(self, no_sleep):
        llm = _FlakyLLM([Exception("429")] * 6)
        with pytest.raises(Exception, match="429"):
            nodes._invoke_with_retry(llm, [])
        assert llm.calls == 5

    def test_404_fails_fast_without_retrying(self, no_sleep):
        """A 404 (e.g. wrong deployment name) won't fix itself, so don't wait and retry."""
        llm = _FlakyLLM([Exception("HTTP 404: DeploymentNotFound")] * 3)
        with pytest.raises(Exception, match="404"):
            nodes._invoke_with_retry(llm, [])
        assert llm.calls == 1
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
