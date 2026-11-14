import json

import pytest

from app.agents import nodes


def _classifier_reply(intent="RESEARCH", academic=False, web=True):
    return json.dumps(
        {"intent": intent, "requires_academic": academic, "requires_web": web, "reasoning": "r"}
    )


@pytest.fixture
def fake_llm(monkeypatch, llm_response):
    """Patch the LLM call; `fake_llm.reply` is returned, prompts are recorded."""

    class Handle:
        reply = _classifier_reply()
        calls = 0
        prompts = []

    def _invoke(state, prompt_messages):
        Handle.calls += 1
        Handle.prompts.append(prompt_messages)
        return llm_response(Handle.reply)

    Handle.calls = 0
    Handle.prompts = []
    monkeypatch.setattr(nodes, "_invoke_with_llm", _invoke)
    return Handle


class TestIntentClassifier:
    def test_research_intent(self, fake_llm):
        fake_llm.reply = _classifier_reply("RESEARCH", academic=True)
        out = nodes.intent_classifier_node({"query": "Latest LLM architectures", "messages": []})
        assert out["intent"] == "RESEARCH"
        assert out["requires_academic"] is True
        assert out["current_step"] == "planning"

    def test_chat_intent_routes_to_chat(self, fake_llm):
        fake_llm.reply = _classifier_reply("CHAT")
        out = nodes.intent_classifier_node({"query": "What is a neural network?", "messages": []})
        assert out["intent"] == "CHAT"
        assert out["current_step"] == "chat"

    def test_rejected_query_goes_to_chat_with_rejected_intent(self, fake_llm):
        fake_llm.reply = _classifier_reply("REJECTED", web=False)
        out = nodes.intent_classifier_node({"query": "How to bake pizza?", "messages": []})
        assert out == {"intent": "REJECTED", "current_step": "chat"}

    def test_bare_rejected_reply_is_tolerated(self, fake_llm):
        fake_llm.reply = "REJECTED"
        out = nodes.intent_classifier_node({"query": "How to bake pizza?", "messages": []})
        assert out == {"intent": "REJECTED", "current_step": "chat"}

    def test_word_rejected_in_reasoning_does_not_reject(self, fake_llm):
        """Regression: any reply containing 'rejected' used to be treated as off-topic."""
        fake_llm.reply = json.dumps(
            {
                "intent": "RESEARCH",
                "requires_academic": False,
                "requires_web": True,
                "reasoning": "AI question, so it is not REJECTED",
            }
        )
        out = nodes.intent_classifier_node({"query": "Why are some LLM papers rejected?", "messages": []})
        assert out["intent"] == "RESEARCH"
        assert out["current_step"] == "planning"

    def test_research_mode_skips_the_llm(self, fake_llm):
        out = nodes.intent_classifier_node({"query": "anything", "mode": "research", "messages": []})
        assert fake_llm.calls == 0
        assert out["intent"] == "RESEARCH"
        assert out["requires_web"] is True
        assert out["requires_academic"] is True
        assert out["current_step"] == "planning"

    def test_auto_mode_calls_the_llm(self, fake_llm):
        nodes.intent_classifier_node({"query": "q", "mode": "auto", "messages": []})
        assert fake_llm.calls == 1

    def test_unparseable_output_falls_back_to_research(self, fake_llm):
        fake_llm.reply = "this is not json"
        out = nodes.intent_classifier_node({"query": "q", "messages": []})
        assert out["intent"] == "RESEARCH"
        assert out["current_step"] == "planning"
        assert "Intent classification error" in out["error"]

    def test_conversation_context_is_in_prompt(self, fake_llm, prompt_text):
        nodes.intent_classifier_node(
            {"query": "and its cost?", "messages": [{"role": "user", "content": "about LoRA"}]}
        )
        assert "about LoRA" in prompt_text(fake_llm.prompts[0])


class TestPlanner:
    def test_parses_steps(self, fake_llm):
        fake_llm.reply = json.dumps(
            {"steps": [{"query": "lora basics", "type": "web"}, {"query": "lora paper", "type": "academic"}]}
        )
        out = nodes.planner_node({"query": "lora", "messages": []})
        assert out["current_step"] == "research"
        assert [s["type"] for s in out["plan"]] == ["web", "academic"]

    def test_failure_falls_back_to_single_web_step(self, fake_llm):
        fake_llm.reply = "garbage"
        out = nodes.planner_node({"query": "lora", "messages": []})
        assert out["plan"] == [{"query": "lora", "type": "web"}]
        assert "Planning error" in out["error"]
