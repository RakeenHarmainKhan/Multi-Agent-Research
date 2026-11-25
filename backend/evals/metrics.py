"""Pure metric functions for the evaluation: no LLM, no network."""
import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

_SENTENCE_BREAK = re.compile(r"(?<=[.!?\]])(?<!\s[A-Z]\.)(?<!^[A-Z]\.)\s+(?=[A-Z])")
_MARKER = re.compile(r"\[(\d+)\]")


def claim_sentences(report: str) -> List[str]:
    """Sentences of the report body. Headings and short fragments are not claims."""
    sentences: List[str] = []
    for line in report.splitlines():
        line = line.strip().lstrip("-*• ").strip()
        if not line or line.startswith("#"):
            continue
        sentences += [s.strip() for s in _SENTENCE_BREAK.split(line) if len(s.split()) >= 4]
    return sentences


def cited_ids(sentence: str) -> List[int]:
    return [int(n) for n in _MARKER.findall(sentence)]


def citation_coverage(report: str) -> Tuple[int, int]:
    """(claim sentences carrying at least one [n], all claim sentences)."""
    sentences = claim_sentences(report)
    return sum(1 for s in sentences if cited_ids(s)), len(sentences)


def structure_ok(report: str) -> bool:
    """The sections the writer prompt asks for."""
    return "## Executive Summary" in report and "## Key Findings" in report


def word_count(report: str) -> int:
    return len(report.split())


def source_stats(sources: List[Dict[str, Any]], citations: List[Dict[str, Any]]) -> Dict[str, int]:
    domains = {urlparse(s["url"]).netloc for s in sources if s.get("url")}
    return {
        "total": len(sources),
        "web": sum(1 for s in sources if s.get("type") == "web"),
        "academic": sum(1 for s in sources if s.get("type") == "academic"),
        "domains": len(domains),
        "cited": len(citations),
    }


def percentile(values: List[float], p: float) -> float:
    """Nearest-rank percentile."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]


def score_band(score: float) -> str:
    """Tavily score bucket, to see where relevant and irrelevant results separate."""
    return "<0.5" if score < 0.5 else "0.5-0.7" if score < 0.7 else ">=0.7"


def rate(hits: int, total: int) -> float:
    return round(hits / total, 3) if total else 0.0


def confusion(pairs: List[Tuple[str, str]]) -> Dict[str, Dict[str, int]]:
    """{expected: {predicted: count}} from (expected, predicted) pairs."""
    table: Dict[str, Dict[str, int]] = {}
    for expected, predicted in pairs:
        row = table.setdefault(expected, {})
        row[predicted] = row.get(predicted, 0) + 1
    return table


def parse_json(text: str) -> Optional[Dict[str, Any]]:
    """The first JSON object in a model reply, or None."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None
