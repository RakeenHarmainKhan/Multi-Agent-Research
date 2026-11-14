"""
Research tools for agents to interact with external data sources.
Implements web search, academic search, and citation extraction.
"""
import logging
import re
from typing import Any, Dict, List, Tuple

from langchain_community.utilities import ArxivAPIWrapper
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from app.core.config import settings

logger = logging.getLogger(__name__)


class _MockSearchTool:
    """Fallback search tool that keeps the workflow functional without a Tavily key."""

    def invoke(self, query: str) -> List[Dict[str, Any]]:
        return [
            {
                "title": "Search API not configured",
                "url": "",
                "content": (
                    "Web search is currently running in fallback mode because no "
                    "TAVILY_API_KEY is configured. "
                    f"Original query: {query}"
                ),
            }
        ]


class ResearchToolkit:
    """Collection of research tools with fallback mechanisms."""

    def __init__(self):
        self.tavily_available = bool(settings.TAVILY_API_KEY)

    def get_web_search_tool(self):
        """Return the Tavily web search tool, or a mock when no key is configured."""
        if not self.tavily_available:
            logger.warning("No web search API configured, using mock data")
            return _MockSearchTool()
        try:
            return TavilySearch(
                tavily_api_key=settings.TAVILY_API_KEY,
                max_results=3,
                search_depth="basic",
                include_raw_content=False,
                include_answer=True,
                exclude_domains=["facebook.com", "instagram.com", "twitter.com", "youtube.com", "tiktok.com"],
            )
        except Exception:
            logger.info("Tavily search API key missing or invalid, using mock fallback.")
            return _MockSearchTool()


@tool
def academic_search(query: str, max_results: int = 3) -> str:
    """
    Search academic databases (Arxiv) for peer-reviewed papers.

    Args:
        query: Scientific query string
        max_results: Maximum number of papers to return

    Returns:
        Formatted string with paper titles, abstracts, and URLs
    """
    try:
        arxiv = ArxivAPIWrapper(
            top_k_results=max_results,
            doc_content_chars_max=2000,
            load_max_docs=3,
            sort_by="relevance",
        )
        return arxiv.run(query)
    except Exception as e:
        logger.error("Academic search failed: %s", e)
        return f"Academic search unavailable: {e}"


def number_sources(
    web_results: List[Dict[str, Any]], academic_results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Merge web and academic results into one list with stable 1-based ids."""
    sources: List[Dict[str, Any]] = []
    for item in web_results:
        sources.append({**item, "id": len(sources) + 1, "type": "web"})
    for item in academic_results:
        sources.append({**item, "id": len(sources) + 1, "type": "academic", "title": item.get("title", "Academic paper")})
    return sources


_CITATION_PATTERN = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def build_citations(report: str, sources: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Validate [n] markers in the report against the numbered sources.

    Markers that point to a non-existent source are removed from the report, so
    every remaining [n] resolves to a real retrieved source. Returns the cleaned
    report and the cited sources in order of first appearance.
    """
    by_id = {source["id"]: source for source in sources}
    cited_ids: List[int] = []

    def _validate(match: re.Match) -> str:
        ids = [int(i) for i in match.group(1).split(",")]
        valid = [i for i in ids if i in by_id]
        if len(valid) != len(ids):
            logger.warning("Dropped invalid citation ids %s", [i for i in ids if i not in by_id])
        for i in valid:
            if i not in cited_ids:
                cited_ids.append(i)
        return "".join(f"[{i}]" for i in valid)

    cleaned = _CITATION_PATTERN.sub(_validate, report)
    citations = [
        {
            "id": i,
            "type": by_id[i]["type"],
            "title": by_id[i].get("title", ""),
            "url": by_id[i].get("url", ""),
            "snippet": by_id[i].get("content", "")[:300],
        }
        for i in cited_ids
    ]
    return cleaned, citations
