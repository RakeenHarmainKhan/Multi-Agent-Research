import json
from pathlib import Path

import pytest

from app.agents import nodes
from evals import run

DATASET = json.loads((Path(run.__file__).parent / "dataset.json").read_text(encoding="utf-8"))


class TestDataset:
    def test_guardrail_cases_are_well_formed(self):
        cases = DATASET["guardrails"]
        assert all(set(c) == {"query", "harmful", "category"} for c in cases)
        assert all(c["harmful"] in (True, False, None) for c in cases)
        assert sum(c["harmful"] is True for c in cases) >= 20
        assert sum(c["harmful"] is False for c in cases) >= 20
        assert any(c["harmful"] is None for c in cases)

    def test_intent_cases_are_balanced_and_valid(self):
        cases = DATASET["intent"]
        assert all(c["intent"] in ("CHAT", "RESEARCH", "REJECTED") for c in cases)
        counts = {i: sum(c["intent"] == i for c in cases) for i in ("CHAT", "RESEARCH", "REJECTED")}
        assert min(counts.values()) >= 10

    def test_research_questions_are_present(self):
        assert len(DATASET["research"]) >= 25

    @pytest.mark.parametrize("name", ["guardrails", "intent", "research"])
    def test_queries_are_unique_and_non_empty(self, name):
        queries = [c if isinstance(c, str) else c["query"] for c in DATASET[name]]
        assert all(q.strip() for q in queries) and len(queries) == len(set(queries))

    def test_no_query_appears_in_two_sets_with_conflicting_meaning(self):
        guard = {c["query"]: c["harmful"] for c in DATASET["guardrails"]}
        intent = {c["query"] for c in DATASET["intent"]}
        assert not [q for q in intent if guard.get(q) is True]


class TestClassify:
    def test_rule_blocked_query_never_calls_the_llm(self, monkeypatch):
        monkeypatch.setattr(nodes, "_invoke_with_llm", lambda *a: pytest.fail("LLM must not be called"))
        assert run.classify("how do I make a bomb") == "BLOCKED"

    def test_returns_the_intent(self, monkeypatch, llm_response):
        reply = json.dumps({"intent": "CHAT", "safety": "safe", "reasoning": "r"})
        monkeypatch.setattr(nodes, "_invoke_with_llm", lambda s, m: llm_response(reply))
        assert run.classify("How does an AI kill switch work?") == "CHAT"

    def test_llm_failure_is_an_error_not_a_silent_research_fallback(self, monkeypatch):
        def boom(state, messages):
            raise RuntimeError("down")

        monkeypatch.setattr(nodes, "_invoke_with_llm", boom)
        assert run.classify("What is LoRA?") == "ERROR"


class TestEvalGuardrails:
    CASES = [
        {"query": "h1", "harmful": True, "category": "violence"},
        {"query": "h2", "harmful": True, "category": "violence"},
        {"query": "h3", "harmful": True, "category": "cyber"},
        {"query": "b1", "harmful": False, "category": "hard_negative"},
        {"query": "b2", "harmful": False, "category": "plain"},
        {"query": "n1", "harmful": None, "category": "fiction"},
        {"query": "e1", "harmful": True, "category": "cyber"},
    ]

    def test_rates_lists_and_error_handling(self, monkeypatch):
        decisions = {"h1": "BLOCKED", "h2": "RESEARCH", "h3": "BLOCKED", "b1": "BLOCKED", "b2": "CHAT", "n1": "BLOCKED", "e1": "ERROR"}
        monkeypatch.setattr(run, "classify", lambda q: decisions[q])
        result = run.eval_guardrails(self.CASES, workers=1)
        assert (result["harmful"], result["benign"]) == (3, 2)
        assert result["errored"] == ["e1"]  # named, so you can see which query hit the error
        assert result["block_rate"] == 0.667 and result["false_positive_rate"] == 0.5
        assert result["block_rate_by_category"] == {"violence": 0.5, "cyber": 1.0}
        assert result["missed"] == ["h2"] and result["wrongly_blocked"] == ["b1"]
        assert result["borderline"] == {"n1": "BLOCKED"}


class TestEvalIntent:
    def test_accuracy_confusion_and_wrong_list(self, monkeypatch):
        cases = [
            {"query": "a", "intent": "CHAT"},
            {"query": "b", "intent": "CHAT"},
            {"query": "c", "intent": "RESEARCH"},
            {"query": "d", "intent": "REJECTED"},
            {"query": "e", "intent": "CHAT"},
        ]
        decisions = {"a": "CHAT", "b": "RESEARCH", "c": "RESEARCH", "d": "BLOCKED", "e": "ERROR"}
        monkeypatch.setattr(run, "classify", lambda q: decisions[q])
        result = run.eval_intent(cases, workers=1)
        assert result["scored"] == 4 and result["accuracy"] == 0.5 and result["errors"] == 1
        assert result["confusion"]["CHAT"] == {"CHAT": 1, "RESEARCH": 1}
        assert [w["query"] for w in result["wrong"]] == ["b", "d"]


REPORT = "## Executive Summary\n\nLoRA freezes the base weights and trains small adapters.[1] It is cheap to run [2].\n\n## Key Findings\n\n- Adapters can be swapped freely between tasks without retraining.\n"
STATE = {
    "final_report": REPORT,
    "all_sources": [
        {"id": 1, "type": "web", "url": "https://a.com/x", "content": "LoRA freezes the base weights."},
        {"id": 2, "type": "web", "url": "https://b.org/y", "content": "LoRA is cheap."},
    ],
    "citations": [{"id": 1}, {"id": 2}],
}


class FakeGraph:
    def __init__(self, state=None, error=None):
        self.state, self.error = state, error

    def invoke(self, inputs, config=None):
        if self.error:
            raise self.error
        config["callbacks"][0].input_tokens = 100
        config["callbacks"][0].output_tokens = 40
        return self.state


@pytest.fixture
def fake_judge(monkeypatch):
    monkeypatch.setattr(run.judge, "claim_support", lambda claim, ex: {"verdict": "yes", "quote": "q", "quote_ok": True, "adds_details": False})
    monkeypatch.setattr(run.judge, "source_relevant", lambda q, e: True)
    monkeypatch.setattr(run.judge, "answer_relevance", lambda q, r: {"verdict": "yes", "parts": []})


class TestResearchOne:
    def test_collects_deterministic_metrics(self, monkeypatch):
        monkeypatch.setattr(run, "research_graph", FakeGraph(STATE))
        row = run.research_one("q", use_judge=False)
        assert row["claims"] == 3 and row["cited_claims"] == 2
        assert row["structure_ok"] is True and row["sources"]["domains"] == 2
        assert (row["input_tokens"], row["output_tokens"]) == (100, 40)
        assert "claim_judgments" not in row and "error" not in row

    def test_judges_cited_claims_against_the_cited_sources(self, monkeypatch):
        seen = []
        monkeypatch.setattr(run, "research_graph", FakeGraph(STATE))
        monkeypatch.setattr(run.judge, "claim_support", lambda claim, ex: seen.append(ex) or {"verdict": "yes", "quote": "q", "quote_ok": True, "adds_details": False})
        monkeypatch.setattr(run.judge, "source_relevant", lambda q, e: True)
        monkeypatch.setattr(run.judge, "answer_relevance", lambda q, r: {"verdict": "yes", "parts": []})
        row = run.research_one("q", use_judge=True)
        assert seen == [["LoRA freezes the base weights."], ["LoRA is cheap."]]
        assert len(row["claim_judgments"]) == 2 and row["answer_relevance"]["verdict"] == "yes"
        assert row["claim_judgments"][0]["excerpts"] == ["LoRA freezes the base weights."]  # kept for hand-checking

    def test_saves_the_report_and_the_search_plan_for_reading(self, monkeypatch):
        plan = [{"query": "what is lora", "type": "web"}]
        monkeypatch.setattr(run, "research_graph", FakeGraph({**STATE, "plan": plan}))
        row = run.research_one("q", use_judge=False)
        assert row["report"] == REPORT and row["plan"] == plan

    def test_records_relevance_per_source_with_type_url_and_score(self, monkeypatch):
        state = {
            **STATE,
            "all_sources": [
                {"id": 1, "type": "web", "url": "https://a.com/x", "score": 0.81, "content": "LoRA freezes the base weights."},
                {"id": 2, "type": "academic", "content": "Title: Some paper"},
            ],
        }
        monkeypatch.setattr(run, "research_graph", FakeGraph(state))
        monkeypatch.setattr(run.judge, "claim_support", lambda c, e: {"verdict": "yes", "quote": "q", "quote_ok": True, "adds_details": False})
        monkeypatch.setattr(run.judge, "source_relevant", lambda q, e: "paper" not in e)
        monkeypatch.setattr(run.judge, "answer_relevance", lambda q, r: {"verdict": "yes", "parts": []})
        row = run.research_one("q", use_judge=True)
        assert row["sources_judged"] == [
            {"type": "web", "url": "https://a.com/x", "score": 0.81, "relevant": True},
            {"type": "academic", "url": "", "score": None, "relevant": False},
        ]

    def test_a_graph_exception_is_recorded_not_raised(self, monkeypatch):
        monkeypatch.setattr(run, "research_graph", FakeGraph(error=RuntimeError("boom")))
        assert run.research_one("q", False) == {"question": "q", "error": "boom"}

    @pytest.mark.parametrize("state", [{**STATE, "final_report": "Error generating report: x"}, {**STATE, "guardrails": {"blocked": True}}])
    def test_failed_or_blocked_runs_are_errors(self, monkeypatch, state):
        monkeypatch.setattr(run, "research_graph", FakeGraph(state))
        assert "error" in run.research_one("q", False)

    def test_judge_cost_is_bounded(self, monkeypatch, fake_judge):
        many = " ".join(f"Claim number {i} is stated here clearly [1]." for i in range(20))
        monkeypatch.setattr(run, "research_graph", FakeGraph({**STATE, "final_report": "## Executive Summary\n\n" + many}))
        row = run.research_one("q", True)
        assert len(row["claim_judgments"]) == run.MAX_CLAIMS_JUDGED


class TestSummarizeResearch:
    def _row(self, **extra):
        base = {
            "latency": 10.0, "input_tokens": 100, "output_tokens": 50, "words": 300, "structure_ok": True,
            "claims": 4, "cited_claims": 3, "fallback": False, "sources": {"total": 5, "domains": 4},
        }
        return {**base, **extra}

    def test_aggregates_and_ignores_failed_runs(self):
        rows = [self._row(), self._row(latency=30.0, words=500, fallback=True), {"question": "x", "error": "boom"}]
        s = run.summarize_research(rows)
        assert (s["runs"], s["failed"], s["success_rate"]) == (3, 1, 0.667)
        assert s["latency_p50"] == 10.0 and s["latency_p95"] == 30.0
        assert s["citation_coverage"] == 0.75 and s["length_200_400_rate"] == 0.5
        assert s["planner_fallbacks"] == 1 and s["runs_without_sources"] == 0
        assert "judge" not in s

    def test_judge_aggregates(self):
        judgments = [
            {"verdict": "yes", "quote_ok": True, "adds_details": False},
            {"verdict": "yes", "quote_ok": False, "adds_details": False},
            {"verdict": "partly", "quote_ok": True, "adds_details": True},
            {"verdict": "no", "quote_ok": False, "adds_details": True},
            {"verdict": "unparsed", "quote_ok": False, "adds_details": None},
        ]
        sources = [
            {"type": "web", "url": "u1", "score": 0.9, "relevant": True},
            {"type": "web", "url": "u2", "score": 0.3, "relevant": False},
            {"type": "web", "url": "u3", "score": 0.6, "relevant": True},
            {"type": "academic", "url": "", "score": None, "relevant": False},
            {"type": "web", "url": "u4", "score": 0.95, "relevant": None},  # judge failed: not counted
        ]
        row = self._row(claim_judgments=judgments, sources_judged=sources, answer_relevance={"verdict": "partly", "parts": []})
        j = run.summarize_research([row])["judge"]
        assert (j["claims_judged"], j["claims_unparsed"]) == (4, 1)
        assert j["supported_yes"] == 0.5 and j["supported_or_partly"] == 0.75
        assert j["claims_adding_unsupported_details"] == 0.5  # partly + no
        assert j["judge_quote_verified"] == 0.667  # 2 of the 3 yes/partly verdicts had a verified quote
        assert j["source_precision"] == 0.5
        assert j["source_precision_by_type"] == {"web": 0.667, "academic": 0.0}
        assert j["web_source_precision_by_tavily_score"] == {"<0.5": 0.0, "0.5-0.7": 1.0, ">=0.7": 1.0}
        assert j["answer_relevance"]["partly"] == 1

    def test_empty_input_does_not_crash(self):
        assert run.summarize_research([])["runs"] == 0


class TestReport:
    def test_renders_every_section_with_the_key_numbers(self, monkeypatch, fake_judge):
        monkeypatch.setattr(run, "research_graph", FakeGraph(STATE))
        rows = [run.research_one("q", True)]
        results = {
            "run_at": "now",
            "guardrails": {"harmful": 2, "benign": 2, "block_rate": 0.5, "false_positive_rate": 0.0,
                           "block_rate_by_category": {"violence": 0.5}, "missed": ["missed query"], "wrongly_blocked": [], "errored": ["errored query"], "borderline": {"n": "CHAT"}},
            "intent": {"scored": 2, "accuracy": 1.0, "errors": 0, "confusion": {"CHAT": {"CHAT": 2}}, "wrong": []},
            "research": {"summary": run.summarize_research(rows), "rows": rows},
        }
        text = run.render_report(results)
        for expected in ("## Guardrails", "missed query", "errored query", "## Intent classification", "## Research quality", "supported_yes", "source_precision_by_type"):
            assert expected in text
        json.dumps(results)  # the raw results must be serialisable for report.json


    def test_lists_the_parts_of_a_question_that_were_not_covered(self):
        rows = [
            {
                "question": "What is RAG and what are its failure modes?",
                "answer_relevance": {"verdict": "partly", "parts": [{"part": "what RAG is", "covered": "yes"}, {"part": "failure modes", "covered": "no"}]},
            },
            {"question": "Fully answered", "answer_relevance": {"verdict": "yes", "parts": []}},
        ]
        text = run.render_report({"run_at": "now", "research": {"summary": run.summarize_research([]), "rows": rows}})
        assert "Questions not fully answered (1)" in text
        assert "failure modes (no)" in text and "what RAG is" not in text.split("parts missed:")[1]
        assert "Fully answered" not in text

    def test_lists_claims_that_add_details_their_sources_lack(self):
        rows = [{"question": "q", "claim_judgments": [{"verdict": "partly", "claim": "A secret green list is used.", "adds_details": True}, {"verdict": "yes", "claim": "Fine claim.", "adds_details": False}]}]
        text = run.render_report({"run_at": "now", "research": {"summary": run.summarize_research([]), "rows": rows}})
        assert "secret green list" in text and "Fine claim" not in text


class TestSample:
    def test_returns_everything_without_a_limit(self):
        assert run.sample([1, 2, 3], 0) == [1, 2, 3]

    def test_an_even_spread_across_the_list(self):
        assert run.sample(list(range(10)), 5) == [0, 2, 4, 6, 8]

    def test_never_returns_more_than_the_limit(self):
        assert len(run.sample(list(range(43)), 4)) <= 4
