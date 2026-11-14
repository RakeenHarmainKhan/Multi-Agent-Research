from app.tools.research import build_citations, number_sources


def _sources():
    return number_sources(
        [
            {"title": "A", "url": "http://a", "content": "alpha"},
            {"title": "B", "url": "http://b", "content": "beta"},
        ],
        [{"content": "Published: 2025\nTitle: Paper X\nAuthors: Z", "type": "academic"}],
    )


class TestNumberSources:
    def test_ids_are_sequential_across_web_and_academic(self):
        assert [s["id"] for s in _sources()] == [1, 2, 3]

    def test_types_are_assigned(self):
        assert [s["type"] for s in _sources()] == ["web", "web", "academic"]

    def test_academic_title_is_parsed(self):
        assert _sources()[2]["title"] == "Paper X"

    def test_academic_title_fallback(self):
        sources = number_sources([], [{"content": "no title line here"}])
        assert sources[0]["title"] == "Academic paper"

    def test_empty_inputs(self):
        assert number_sources([], []) == []

    def test_does_not_mutate_inputs(self):
        web = [{"title": "A", "url": "http://a", "content": "x"}]
        number_sources(web, [])
        assert "id" not in web[0]


class TestBuildCitations:
    def test_valid_marker_resolves_to_source(self):
        report, citations = build_citations("Fact [2].", _sources())
        assert report == "Fact [2]."
        assert citations == [
            {"id": 2, "type": "web", "title": "B", "url": "http://b", "snippet": "beta"}
        ]

    def test_invalid_marker_is_stripped(self):
        report, citations = build_citations("Fake claim [9].", _sources())
        assert "[9]" not in report
        assert citations == []

    def test_multi_id_marker_is_normalised(self):
        report, citations = build_citations("Both [1, 3].", _sources())
        assert report == "Both [1][3]."
        assert [c["id"] for c in citations] == [1, 3]

    def test_partially_invalid_multi_marker_keeps_valid_ids(self):
        report, citations = build_citations("Mixed [1, 9].", _sources())
        assert report == "Mixed [1]."
        assert [c["id"] for c in citations] == [1]

    def test_order_is_first_appearance_and_deduplicated(self):
        _, citations = build_citations("x [2] y [1] z [2]", _sources())
        assert [c["id"] for c in citations] == [2, 1]

    def test_non_numeric_brackets_are_untouched(self):
        report, citations = build_citations("See [Note] and [a](http://x).", _sources())
        assert report == "See [Note] and [a](http://x)."
        assert citations == []

    def test_no_sources_strips_all_markers(self):
        report, citations = build_citations("Claim [1].", [])
        assert "[1]" not in report
        assert citations == []

    def test_snippet_is_truncated(self):
        sources = number_sources([{"title": "L", "url": "u", "content": "x" * 1000}], [])
        _, citations = build_citations("[1]", sources)
        assert len(citations[0]["snippet"]) == 300

    def test_academic_citation_has_empty_url(self):
        _, citations = build_citations("[3]", _sources())
        assert citations[0]["url"] == ""
        assert citations[0]["type"] == "academic"
