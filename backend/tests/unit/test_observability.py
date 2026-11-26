"""Offline tracing checks using the real SDK and an in-memory exporter."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langfuse import Langfuse
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.agents import nodes
from app.agents.workflow import _timed_node
from app.api.models import ResearchRequest
from app.api.routes import execute_research
from app.observability import tracing


@pytest.fixture
def telemetry(monkeypatch):
    def deny_network(request):
        raise AssertionError("Tracing tests must not use the network")

    exporter = InMemorySpanExporter()
    client = Langfuse(
        public_key=f"pk-lf-{uuid4().hex}",
        secret_key="test-secret",
        tracer_provider=TracerProvider(),
        span_exporter=exporter,
        httpx_client=httpx.Client(transport=httpx.MockTransport(deny_network)),
    )
    monkeypatch.setattr(tracing, "_client", client)
    yield client, exporter
    client.shutdown()
    assert tracing._current.get() is None


def _metadata(span, key):
    return json.loads(span.attributes[f"langfuse.observation.metadata.{key}"])


@pytest.mark.parametrize("intent", ["CHAT", "LOOKUP", "RESEARCH"])
def test_real_graph_exports_nested_agents_and_exact_tokens_without_payloads(telemetry, monkeypatch, intent):
    client, exporter = telemetry
    responses = [json.dumps({"intent": intent, "search_query": "PRIVATE-SEARCH", "reasoning": "PRIVATE-REASON"})]
    if intent == "RESEARCH":
        responses.append(json.dumps({"steps": [{"query": "PRIVATE-SEARCH", "type": "web"}]}))
    responses.append("PRIVATE-ANSWER [1]")
    if intent == "LOOKUP":
        responses.append(json.dumps({"answered": True}))
    llm = FakeMessagesListChatModel(responses=[
        AIMessage(content=text, usage_metadata={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14})
        for text in responses
    ])
    monkeypatch.setattr(nodes, "_build_llm", lambda: llm)
    tool = SimpleNamespace(invoke=lambda query: {"results": [
        {"title": "PRIVATE-TITLE", "content": "PRIVATE-SOURCE", "url": "https://private.example", "score": 1},
    ]})
    monkeypatch.setattr(nodes, "web_search", tool)
    response = asyncio.run(execute_research(ResearchRequest(query="PRIVATE-QUESTION", thread_id="PRIVATE-THREAD")))
    client.flush()
    spans = exporter.get_finished_spans()
    root = next(span for span in spans if span.name == "research")
    agents = {span.name: span for span in spans if span.parent and span.parent.span_id == root.context.span_id}
    expected = {"input_guard", "intent_classifier"}
    expected |= {"simple_chat"} if intent == "CHAT" else {"researcher", "writer"}
    if intent == "RESEARCH":
        expected.add("planner")
    assert set(agents) == expected
    assert len({span.context.trace_id for span in spans}) == 1
    calls = len(responses)
    assert response.input_tokens == _metadata(root, "input_tokens") == calls * 10
    assert response.output_tokens == _metadata(root, "output_tokens") == calls * 4
    assert _metadata(root, "total_tokens") == calls * 14
    generations = [span for span in spans if span.name == "llm"]
    assert len(generations) == calls
    for generation in generations:
        assert generation.parent.span_id in {span.context.span_id for span in agents.values()}
        assert json.loads(generation.attributes["langfuse.observation.usage_details"]) == {
            "input": 10, "output": 4, "total": 14,
        }
    for agent in agents.values():
        children = [span for span in generations if span.parent.span_id == agent.context.span_id]
        assert _metadata(agent, "total_tokens") == len(children) * 14
        assert _metadata(agent, "latency_seconds") >= 0
    exported = str([dict(span.attributes) for span in spans])
    assert "PRIVATE-" not in exported
    assert "private.example" not in exported


def test_handled_and_raised_errors_are_visible_without_exception_text(telemetry, monkeypatch):
    client, exporter = telemetry

    def failing_llm(state, messages):
        raise ValueError("PRIVATE-PROVIDER-ERROR")

    monkeypatch.setattr(nodes, "_invoke_with_llm", failing_llm)
    with tracing.trace_span("research"):
        result = _timed_node("writer", nodes.writer_node)({"query": "PRIVATE-QUESTION"})
    assert result["error"] == "PRIVATE-PROVIDER-ERROR"
    with pytest.raises(RuntimeError, match="PRIVATE-UNHANDLED"):
        with tracing.trace_span("failed-run"):
            raise RuntimeError("PRIVATE-UNHANDLED")
    client.flush()
    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert _metadata(spans["writer"], "error_type") == "ValueError"
    assert _metadata(spans["research"], "child_error") is True
    assert _metadata(spans["failed-run"], "error_type") == "RuntimeError"
    assert spans["writer"].attributes["langfuse.observation.level"] == "ERROR"
    assert "PRIVATE-" not in str([dict(span.attributes) for span in spans.values()])
    assert not any(span.events for span in spans.values())


@pytest.mark.parametrize("state, reply, status", [
    ({"query": "q"}, {"intent": "CHAT", "safety": "harmful", "reasoning": "r"}, "blocked"),
    ({"query": "q"}, {"intent": "CHAT", "safety": "safe", "reasoning": "r"}, "accepted"),
    ({"query": "q", "mode": "research"}, {"intent": "CHAT", "safety": "harmful", "reasoning": "r"}, "blocked"),
    ({"query": "q"}, "not json", "unchecked"),
    ({"query": "q", "guard_suspect": True}, "not json", "blocked"),
])
def test_classifier_span_records_safety_status(telemetry, monkeypatch, llm_response, state, reply, status):
    client, exporter = telemetry
    text = json.dumps(reply) if isinstance(reply, dict) else reply
    monkeypatch.setattr(nodes, "_invoke_with_llm", lambda s, messages: llm_response(text))
    _timed_node("intent_classifier", nodes.intent_classifier_node)(state)
    client.flush()
    span = next(span for span in exporter.get_finished_spans() if span.name == "intent_classifier")
    assert _metadata(span, "status") == status


def test_retried_llm_attempts_keep_errors_and_successful_usage(telemetry, monkeypatch):
    client, exporter = telemetry
    llm = MagicMock()
    llm.invoke.side_effect = [
        RuntimeError("429 PRIVATE-ERROR"),
        AIMessage(content="PRIVATE-ANSWER", usage_metadata={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}),
    ]
    monkeypatch.setattr(nodes._invoke_with_retry.retry, "sleep", lambda wait: None)
    with tracing.trace_span("research"):
        _timed_node("writer", lambda state: {"answer": nodes._invoke_with_retry(llm, [])})({})
    client.flush()
    spans = exporter.get_finished_spans()
    writer = next(span for span in spans if span.name == "writer")
    assert _metadata(writer, "retry_count") == 1
    assert _metadata(writer, "total_tokens") == 7
    assert _metadata(writer, "child_error") is True
    assert len([span for span in spans if span.name == "llm"]) == 2
    assert "PRIVATE-" not in str([dict(span.attributes) for span in spans])


def test_concurrent_runs_keep_context_and_usage_separate(telemetry):
    client, exporter = telemetry

    async def run(count):
        with tracing.trace_span(f"run-{count}"):
            await asyncio.sleep(0)
            with tracing.trace_span("llm", generation=True) as metadata:
                metadata.update(input_tokens=count, output_tokens=1)

    async def both():
        await asyncio.gather(run(2), run(8))

    asyncio.run(both())
    client.flush()
    roots = [span for span in exporter.get_finished_spans() if span.name.startswith("run-")]
    assert len({span.context.trace_id for span in roots}) == 2
    assert {_metadata(span, "total_tokens") for span in roots} == {3, 9}


def test_disabled_and_missing_configuration(monkeypatch):
    factory = MagicMock()
    monkeypatch.setattr(tracing, "Langfuse", factory)
    monkeypatch.setattr(tracing, "_client", None)
    monkeypatch.setattr(tracing.settings, "LANGFUSE_ENABLED", False)
    tracing.start_tracing()
    with tracing.trace_span("disabled"):
        pass
    factory.assert_not_called()
    monkeypatch.setattr(tracing.settings, "LANGFUSE_ENABLED", True)
    monkeypatch.setattr(tracing.settings, "LANGFUSE_PUBLIC_KEY", "")
    with pytest.raises(ValueError, match="requires"):
        tracing.start_tracing()


@pytest.mark.parametrize("operation", ["initialize", "start", "update", "end", "shutdown"])
def test_sdk_failures_are_logged_without_breaking_execution(monkeypatch, caplog, operation):
    client = MagicMock()
    manager = client.start_as_current_observation.return_value
    failing = {
        "initialize": client,
        "start": manager.__enter__,
        "update": manager.__enter__.return_value.update,
        "end": manager.__exit__,
        "shutdown": client.shutdown,
    }[operation]
    failing.side_effect = RuntimeError("PRIVATE-SDK-ERROR")
    monkeypatch.setattr(tracing, "_client", None if operation == "initialize" else client)
    monkeypatch.setattr(tracing, "Langfuse", client)
    monkeypatch.setattr(tracing.settings, "LANGFUSE_ENABLED", True)
    monkeypatch.setattr(tracing.settings, "LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setattr(tracing.settings, "LANGFUSE_SECRET_KEY", "sk-test")
    if operation == "initialize":
        tracing.start_tracing()
    with tracing.trace_span("run"):
        result = 42
    tracing.stop_tracing()
    assert result == 42
    assert "failed (RuntimeError)" in caplog.text
    assert "PRIVATE-SDK-ERROR" not in caplog.text
    assert tracing._current.get() is None
    if operation == "update":
        manager.__exit__.assert_called_once_with(None, None, None)
