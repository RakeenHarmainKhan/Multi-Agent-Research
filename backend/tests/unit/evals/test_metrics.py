import pytest

from evals import metrics

REPORT = """## Executive Summary

LoRA freezes the base weights and trains small adapters.[1] This cuts trainable parameters a lot [2].

## Key Findings

- **Cost:** Training a 7B model with LoRA takes a few hours on one A100. [1]
- Short fragment here.
- Adapters can be swapped without duplicating the base model, though quality may vary slightly.
"""


class TestClaimSentences:
    def test_headings_and_short_fragments_are_not_claims(self):
        claims = metrics.claim_sentences(REPORT)
        assert not any(c.startswith("#") for c in claims)
        assert "Short fragment here." not in claims

    def test_a_citation_right_after_the_period_ends_the_sentence(self):
        claims = metrics.claim_sentences(REPORT)
        assert claims[0] == "LoRA freezes the base weights and trains small adapters.[1]"
        assert claims[1].startswith("This cuts trainable parameters")

    def test_bullet_markers_and_bold_are_stripped(self):
        assert any(c.startswith("Cost:**") or c.startswith("Cost:") for c in metrics.claim_sentences(REPORT))

    def test_empty_report(self):
        assert metrics.claim_sentences("") == []

    def test_initials_do_not_split_a_cited_claim(self):
        claim = "The most recent prize went to John J. Hopfield and Geoffrey Hinton. [2]"
        following = "The prize was awarded for foundational discoveries. [1]"
        assert metrics.claim_sentences(f"{claim} {following}") == [claim, following]
        assert metrics.citation_coverage(f"{claim} {following}") == (2, 2)

    def test_sentence_after_us_abbreviation_is_still_separate(self):
        first = "The prize was awarded in the U.S."
        second = "The next prize was awarded later. [1]"
        assert metrics.claim_sentences(f"{first} {second}") == [first, second]


class TestCitationCoverage:
    def test_counts_cited_claims_over_all_claims(self):
        cited, total = metrics.citation_coverage(REPORT)
        assert (cited, total) == (3, 4)  # the last bullet has no marker

    def test_no_claims(self):
        assert metrics.citation_coverage("## Heading only") == (0, 0)

    def test_cited_ids_reads_every_marker(self):
        assert metrics.cited_ids("Claim [1][3] and [2].") == [1, 3, 2]
        assert metrics.cited_ids("no markers") == []


class TestSimpleMetrics:
    def test_structure_requires_both_sections(self):
        assert metrics.structure_ok(REPORT)
        assert not metrics.structure_ok("## Executive Summary\nonly one")

    def test_word_count(self):
        assert metrics.word_count("one two  three\nfour") == 4

    def test_source_stats(self):
        sources = [
            {"type": "web", "url": "https://a.com/x"},
            {"type": "web", "url": "https://a.com/y"},
            {"type": "web", "url": "https://b.org/z"},
            {"type": "academic", "url": ""},
        ]
        assert metrics.source_stats(sources, [{"id": 1}]) == {
            "total": 4, "web": 3, "academic": 1, "domains": 2, "cited": 1,
        }

    @pytest.mark.parametrize(
        "values, p, expected",
        [([1, 2, 3, 4, 5], 50, 3), ([5, 1, 3], 50, 3), (list(range(1, 21)), 95, 19), ([7], 95, 7), ([], 50, 0.0)],
    )
    def test_percentile_is_nearest_rank(self, values, p, expected):
        assert metrics.percentile(values, p) == expected

    @pytest.mark.parametrize("score, band", [(0.1, "<0.5"), (0.49, "<0.5"), (0.5, "0.5-0.7"), (0.69, "0.5-0.7"), (0.7, ">=0.7"), (0.95, ">=0.7")])
    def test_score_bands(self, score, band):
        assert metrics.score_band(score) == band

    def test_rate_handles_zero_total(self):
        assert metrics.rate(1, 3) == 0.333 and metrics.rate(0, 0) == 0.0

    def test_confusion_table(self):
        pairs = [("CHAT", "CHAT"), ("CHAT", "RESEARCH"), ("CHAT", "CHAT"), ("REJECTED", "BLOCKED")]
        assert metrics.confusion(pairs) == {"CHAT": {"CHAT": 2, "RESEARCH": 1}, "REJECTED": {"BLOCKED": 1}}


class TestParseJson:
    def test_plain_and_fenced_json(self):
        assert metrics.parse_json('{"a": 1}') == {"a": 1}
        assert metrics.parse_json('Sure!\n```json\n{"verdict": "yes"}\n```') == {"verdict": "yes"}

    @pytest.mark.parametrize("text", ["no json here", "{broken", "[1, 2]", ""])
    def test_anything_else_is_none(self, text):
        assert metrics.parse_json(text) is None
