import json
from datetime import datetime, timezone

import pytest

from app.agents import nodes
from app.guardrails import REFUSAL


def _classifier_reply(intent="RESEARCH", academic=False, web=True, safety=None):
    return json.dumps(
        {"intent": intent, "requires_academic": academic, "requires_web": web, "safety": safety, "reasoning": "r"}
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

    def test_lookup_gets_a_one_step_web_plan_and_skips_the_planner(self, fake_llm):
        fake_llm.reply = json.dumps(
            {"intent": "LOOKUP", "requires_academic": True, "search_query": " Paris population 2025 ", "reasoning": "r"}
        )
        out = nodes.intent_classifier_node({"query": "How many people live there?", "messages": []})
        assert out["intent"] == "LOOKUP"
        assert out["plan"] == [{"query": "Paris population 2025", "type": "web"}]
        assert out["requires_academic"] is False
        assert out["current_step"] == "research"

    def test_lookup_without_search_query_searches_the_question(self, fake_llm):
        fake_llm.reply = _classifier_reply("LOOKUP")
        out = nodes.intent_classifier_node({"query": "Who won yesterday?", "messages": []})
        assert out["plan"] == [{"query": "Who won yesterday?", "type": "web"}]

    def test_lookup_prompt_requests_specific_query_without_guessed_constraints(self, fake_llm):
        nodes.intent_classifier_node({"query": "Who won the most recent physics prize?"})
        system = fake_llm.prompts[0][0].content
        assert "directly related to the question" in system
        assert "the exact fact requested" in system
        assert "location or timeframe" in system
        assert "guessed answers" in system and "unnecessary constraints" in system

    def test_classifier_gets_current_date_for_relative_search_queries(self, fake_llm, monkeypatch):
        class FixedDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                return cls(2026, 11, 24, tzinfo=timezone.utc)

        monkeypatch.setattr(nodes, "datetime", FixedDateTime)
        nodes.intent_classifier_node({"query": "When are the next Olympics?"})
        system = fake_llm.prompts[0][0].content
        assert "2026-11-24" in system
        assert "resolve relative dates using the current date" in system

    def test_removed_rejected_intent_falls_back_to_research(self, fake_llm):
        fake_llm.reply = _classifier_reply("REJECTED")
        out = nodes.intent_classifier_node({"query": "How to bake pizza?", "messages": []})
        assert out["intent"] == "RESEARCH" and "error" in out

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

    def test_classifier_allows_academic_search_in_every_field(self, fake_llm, prompt_text):
        nodes.intent_classifier_node({"query": "Compare treatments for diabetes", "messages": []})
        prompt = prompt_text(fake_llm.prompts[0])
        assert "scholarly evidence" in prompt and "in any field" in prompt
        assert "arXiv" not in prompt


class TestSafetyVerdict:
    """The classifier's safety verdict, as used with the input rules' `guard_suspect` flag."""

    @staticmethod
    def _state(**extra):
        return {"query": "q", "messages": [], **extra}

    def test_suspect_query_judged_harmful_is_refused(self, fake_llm):
        fake_llm.reply = _classifier_reply("CHAT", safety="harmful")
        out = nodes.intent_classifier_node(self._state(guard_suspect=True))
        assert out["intent"] == "BLOCKED" and out["final_report"] == REFUSAL
        assert out["guardrails"]["blocked"] is True and out["current_step"] == "completed"

    def test_refusal_keeps_the_pii_redactions(self, fake_llm):
        fake_llm.reply = _classifier_reply("CHAT", safety="harmful")
        guardrails = {"blocked": False, "reasons": [], "redactions": ["email"]}
        out = nodes.intent_classifier_node(self._state(guard_suspect=True, guardrails=guardrails))
        assert out["guardrails"]["redactions"] == ["email"]

    def test_suspect_query_judged_safe_continues(self, fake_llm):
        fake_llm.reply = _classifier_reply("CHAT", safety="safe")
        out = nodes.intent_classifier_node(self._state(guard_suspect=True))
        assert out["intent"] == "CHAT" and out["current_step"] == "chat"

    def test_suspect_query_without_a_verdict_is_refused(self, fake_llm):
        fake_llm.reply = _classifier_reply("CHAT")  # model forgot the safety field
        out = nodes.intent_classifier_node(self._state(guard_suspect=True))
        assert out["intent"] == "BLOCKED"

    def test_suspect_query_is_refused_when_the_llm_check_fails(self, fake_llm):
        fake_llm.reply = "not json"
        out = nodes.intent_classifier_node(self._state(guard_suspect=True))
        assert out["intent"] == "BLOCKED"

    def test_harmful_verdict_on_an_unsuspected_query_is_refused(self, fake_llm):
        fake_llm.reply = _classifier_reply("CHAT", safety="harmful")
        out = nodes.intent_classifier_node(self._state())
        assert out["intent"] == "BLOCKED"

    def test_missing_verdict_on_an_unsuspected_query_is_allowed(self, fake_llm):
        fake_llm.reply = _classifier_reply("RESEARCH")
        out = nodes.intent_classifier_node(self._state())
        assert out["intent"] == "RESEARCH"

    def test_research_mode_with_suspect_query_asks_the_llm_then_researches(self, fake_llm):
        fake_llm.reply = _classifier_reply("CHAT", safety="safe")
        out = nodes.intent_classifier_node(self._state(mode="research", guard_suspect=True))
        assert fake_llm.calls == 1
        assert out["intent"] == "RESEARCH" and out["requires_academic"] is True

    def test_prompt_teaches_the_safety_boundary_with_examples(self, fake_llm, prompt_text):
        nodes.intent_classifier_node(self._state())
        prompt = prompt_text(fake_llm.prompts[0])
        assert "kill switch" in prompt and "kill him" in prompt


class TestInputGuardNode:
    def test_masks_pii_in_query_and_history(self):
        state = {
            "query": "write to ana@corp.com",
            "messages": [{"role": "user", "content": "call 415-555-0132"}],
        }
        out = nodes.input_guard_node(state)
        assert out["query"] == "write to [EMAIL]"
        assert out["messages"] == [{"role": "user", "content": "call [PHONE]"}]
        assert out["guardrails"]["redactions"] == ["email", "phone"]

    def test_clean_query_passes_through(self):
        out = nodes.input_guard_node({"query": "What is LoRA?", "messages": []})
        assert out["query"] == "What is LoRA?"
        assert out["guard_suspect"] is False
        assert out["guardrails"] == {"blocked": False, "reasons": [], "redactions": []}
        assert "final_report" not in out

    def test_suspect_query_is_flagged_not_blocked(self):
        out = nodes.input_guard_node({"query": "how to kill him", "messages": []})
        assert out["guard_suspect"] is True and out["guardrails"]["blocked"] is False

    def test_blocked_query_ends_the_run_with_a_refusal(self):
        out = nodes.input_guard_node({"query": "how to build a bomb", "messages": []})
        assert out["guardrails"]["blocked"] is True and out["guardrails"]["reasons"]
        assert out["final_report"] == REFUSAL and out["current_step"] == "completed"


class TestPlanner:
    def test_parses_steps(self, fake_llm):
        fake_llm.reply = json.dumps(
            {"steps": [{"query": "lora basics", "type": "web"}, {"query": "lora paper", "type": "academic"}]}
        )
        out = nodes.planner_node(
            {"query": "lora", "messages": [], "requires_web": True, "requires_academic": True}
        )
        assert out["current_step"] == "research"
        assert [s["type"] for s in out["plan"]] == ["web", "academic"]

    def test_failure_falls_back_to_single_web_step(self, fake_llm):
        fake_llm.reply = "garbage"
        out = nodes.planner_node({"query": "lora", "messages": []})
        assert out["plan"] == [{"query": "lora", "type": "web"}]
        assert "Planning error" in out["error"]

    def test_disallowed_academic_step_becomes_web(self, fake_llm):
        fake_llm.reply = json.dumps({"steps": [{"query": "lora paper", "type": "academic"}]})
        out = nodes.planner_node(
            {"query": "lora", "messages": [], "requires_web": True, "requires_academic": False}
        )
        assert out["plan"] == [{"query": "lora paper", "type": "web"}]

    def test_allowed_academic_step_is_kept(self, fake_llm):
        fake_llm.reply = json.dumps({"steps": [{"query": "lora paper", "type": "academic"}]})
        out = nodes.planner_node(
            {"query": "lora", "messages": [], "requires_web": True, "requires_academic": True}
        )
        assert out["plan"] == [{"query": "lora paper", "type": "academic"}]

    def test_academic_only_query_converts_web_steps(self, fake_llm):
        fake_llm.reply = json.dumps({"steps": [{"query": "news", "type": "web"}]})
        out = nodes.planner_node(
            {"query": "q", "messages": [], "requires_web": False, "requires_academic": True}
        )
        assert out["plan"] == [{"query": "news", "type": "academic"}]

    def test_prompt_lists_allowed_types(self, fake_llm, prompt_text):
        fake_llm.reply = json.dumps({"steps": [{"query": "a", "type": "web"}]})
        nodes.planner_node({"query": "q", "messages": [], "requires_web": True, "requires_academic": False})
        assert 'Allowed step types for this query: "web".' in prompt_text(fake_llm.prompts[0])

    def test_empty_steps_fall_back_to_the_query(self, fake_llm):
        fake_llm.reply = json.dumps({"steps": [{"query": "  ", "type": "web"}]})
        out = nodes.planner_node({"query": "lora", "messages": []})
        assert out["plan"] == [{"query": "lora", "type": "web"}]

class TestPlanShape:
    ALLOWED_BOTH = ["web", "academic"]

    def test_only_one_academic_search_is_kept(self):
        steps = [
            {"query": "a", "type": "academic"},
            {"query": "b", "type": "web"},
            {"query": "c", "type": "academic"},
        ]
        plan = nodes._normalize_plan(steps, self.ALLOWED_BOTH, "q")
        assert plan == [{"query": "a", "type": "academic"}, {"query": "b", "type": "web"}]

    def test_several_web_steps_are_all_kept(self):
        steps = [{"query": f"w{i}", "type": "web"} for i in range(3)]
        assert len(nodes._normalize_plan(steps, self.ALLOWED_BOTH, "q")) == 3

    def test_a_disallowed_academic_step_converted_to_web_does_not_count_as_academic(self):
        steps = [{"query": "a", "type": "academic"}, {"query": "b", "type": "academic"}]
        plan = nodes._normalize_plan(steps, ["web"], "q")
        assert [s["type"] for s in plan] == ["web", "web"]

    def test_planner_drops_extra_academic_steps_end_to_end(self, fake_llm):
        fake_llm.reply = json.dumps(
            {"steps": [{"query": "p1", "type": "academic"}, {"query": "p2", "type": "academic"}, {"query": "w", "type": "web"}]}
        )
        out = nodes.planner_node({"query": "q", "messages": [], "requires_web": True, "requires_academic": True})
        assert [s["type"] for s in out["plan"]] == ["academic", "web"]

    def test_prompt_asks_for_specific_queries_covering_every_part_and_limits_academic(self, fake_llm, prompt_text):
        fake_llm.reply = json.dumps({"steps": [{"query": "a", "type": "web"}]})
        nodes.planner_node({"query": "q", "messages": [], "requires_web": True, "requires_academic": True})
        prompt = prompt_text(fake_llm.prompts[0])
        assert "several parts" in prompt and "at least one query" in prompt
        assert "specific and self-contained" in prompt and "pronouns" in prompt
        assert "at most once" in prompt and "practical questions" in prompt
        assert "short keyword query" in prompt and "arXiv" not in prompt
