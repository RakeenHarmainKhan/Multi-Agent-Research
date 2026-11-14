import pytest

from evals import judge


@pytest.fixture
def reply(monkeypatch, llm_response):
    """Make the judge's LLM return a fixed text (or raise)."""

    def set_reply(text=None, error=None):
        def fake(state, messages):
            if error:
                raise error
            return llm_response(text)

        monkeypatch.setattr(judge, "_invoke_with_llm", fake)

    return set_reply


EXCERPTS = ["LoRA freezes the pretrained weights and trains small low-rank matrices."]


class TestClaimSupport:
    def test_supported_claim_with_a_real_quote(self, reply):
        reply('{"verdict": "yes", "quote": "freezes the pretrained weights"}')
        assert judge.claim_support("LoRA freezes weights.", EXCERPTS) == {
            "verdict": "yes", "quote": "freezes the pretrained weights", "quote_ok": True, "adds_details": False,
        }

    @pytest.mark.parametrize("value, expected", [("true", True), ("false", False), ("\"yes\"", None), ("null", None)])
    def test_adds_details_is_a_bool_or_none(self, reply, value, expected):
        reply('{"verdict": "partly", "quote": "x", "adds_details": ' + value + '}')
        assert judge.claim_support("c", EXCERPTS)["adds_details"] is expected

    def test_a_yes_verdict_never_adds_details_even_if_the_judge_contradicts_itself(self, reply):
        reply('{"verdict": "yes", "quote": "freezes the pretrained weights", "adds_details": true}')
        assert judge.claim_support("c", EXCERPTS)["adds_details"] is False

    def test_an_unparsed_verdict_has_no_adds_details(self, reply):
        reply("garbage")
        assert judge.claim_support("c", EXCERPTS)["adds_details"] is None

    def test_quote_match_ignores_case_and_whitespace(self, reply):
        reply('{"verdict": "yes", "quote": "FREEZES  the pretrained\\nweights"}')
        assert judge.claim_support("c", EXCERPTS)["quote_ok"] is True

    def test_a_quote_that_is_not_in_the_excerpts_is_flagged(self, reply):
        """The judge can hallucinate evidence; quote_ok exposes it."""
        reply('{"verdict": "yes", "quote": "LoRA was invented in 1999"}')
        result = judge.claim_support("c", EXCERPTS)
        assert result["verdict"] == "yes" and result["quote_ok"] is False

    def test_a_yes_without_any_quote_is_not_quote_ok(self, reply):
        reply('{"verdict": "yes", "quote": ""}')
        assert judge.claim_support("c", EXCERPTS)["quote_ok"] is False

    @pytest.mark.parametrize("text", ["I think it is supported", '{"verdict": "maybe"}', '{"quote": "x"}'])
    def test_malformed_replies_are_unparsed(self, reply, text):
        reply(text)
        assert judge.claim_support("c", EXCERPTS)["verdict"] == "unparsed"

    def test_an_llm_failure_is_unparsed_not_a_crash(self, reply):
        reply(error=RuntimeError("gateway down"))
        assert judge.claim_support("c", EXCERPTS)["verdict"] == "unparsed"


class TestSourceAndAnswerJudges:
    def test_source_relevance_is_a_bool_or_none(self, reply):
        reply('{"relevant": true}')
        assert judge.source_relevant("q", "e") is True
        reply('{"relevant": "yes"}')
        assert judge.source_relevant("q", "e") is None
        reply(error=RuntimeError("down"))
        assert judge.source_relevant("q", "e") is None

    def test_answer_relevance_returns_the_verdict_and_how_each_part_is_covered(self, reply):
        reply('{"parts": [{"part": "what RAG is", "covered": "yes"}, {"part": "failure modes", "covered": "no"}], "verdict": "partly"}')
        assert judge.answer_relevance("q", "r") == {
            "verdict": "partly",
            "parts": [{"part": "what RAG is", "covered": "yes"}, {"part": "failure modes", "covered": "no"}],
        }

    def test_malformed_parts_are_ignored_but_the_verdict_is_kept(self, reply):
        reply('{"parts": [{"part": "a", "covered": "maybe"}, "junk", {"covered": "yes"}], "verdict": "yes"}')
        assert judge.answer_relevance("q", "r") == {"verdict": "yes", "parts": [{"part": "", "covered": "yes"}]}

    def test_parts_that_is_not_a_list_gives_no_parts(self, reply):
        reply('{"parts": "everything", "verdict": "yes"}')
        assert judge.answer_relevance("q", "r") == {"verdict": "yes", "parts": []}

    def test_unparseable_reply_is_unparsed(self, reply):
        reply("nonsense")
        assert judge.answer_relevance("q", "r") == {"verdict": "unparsed", "parts": []}

    def test_the_prompt_asks_for_each_part_of_the_question(self, monkeypatch, llm_response):
        seen = []
        monkeypatch.setattr(judge, "_invoke_with_llm", lambda s, m: seen.append(m[0].content) or llm_response("{}"))
        judge.answer_relevance("q", "r")
        assert "split the question into its parts" in seen[0].lower()
