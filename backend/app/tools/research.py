"""
Research tools for agents to interact with external data sources.
Implements web search, academic search, and citation extraction.
"""
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_tavily import TavilySearch
from tavily import TavilyClient
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random_exponential

from app.core.config import settings

logger = logging.getLogger(__name__)

_TRANSIENT_ERROR_NAMES = {"ConnectionError", "ConnectTimeout", "ReadTimeout", "Timeout", "RemoteProtocolError"}


def is_transient_search_error(exc: BaseException) -> bool:
    """Network blips, rate limits (429) and server errors (500/502/503/504) are worth retrying; bad input or config is not."""
    if isinstance(exc, (ConnectionError, TimeoutError)) or type(exc).__name__ in _TRANSIENT_ERROR_NAMES:
        return True
    message = str(exc).lower()
    return any(marker in message for marker in ("429", "rate limit", "http 500", "502", "503", "504"))


search_retry = retry(
    stop=stop_after_attempt(5),
    # jitter stops retries from many runs landing together and hitting the rate limit again
    wait=wait_random_exponential(multiplier=3, min=3, max=30),
    retry=retry_if_exception(is_transient_search_error),
    reraise=True,
)


# Academic search is Tavily restricted to scholarly sites: journals, preprint servers and paper indexes across fields.
ACADEMIC_DOMAINS = [
    "arxiv.org", "biorxiv.org", "medrxiv.org", "ssrn.com", "nber.org",
    "pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov", "semanticscholar.org", "openreview.net", "aclanthology.org",
    "nature.com", "science.org", "cell.com", "thelancet.com", "nejm.org", "bmj.com", "jamanetwork.com",
    "plos.org", "frontiersin.org", "pnas.org", "sciencedirect.com", "link.springer.com",
    "onlinelibrary.wiley.com", "tandfonline.com", "academic.oup.com", "cambridge.org", "journals.sagepub.com",
    "jstor.org", "dl.acm.org", "ieeexplore.ieee.org", "iopscience.iop.org",
]


class WebSearchUnavailableError(RuntimeError):
    """Web search is not configured or could not be initialized."""


class _MockSearchTool:
    """Unavailable search tool; do not treat a configuration message as a source."""

    def invoke(self, query: str) -> List[Dict[str, Any]]:
        raise WebSearchUnavailableError("Web search is unavailable")


class ResearchToolkit:
    """Collection of research tools with fallback mechanisms."""

    def __init__(self):
        self.tavily_available = bool(settings.TAVILY_API_KEY)

    def get_web_search_tool(self, include_domains: Optional[List[str]] = None):
        """Return the Tavily search tool (optionally limited to some sites), or a mock when no key is configured."""
        if not self.tavily_available:
            logger.warning("No web search API configured")
            return _MockSearchTool()
        try:
            return TavilySearch(
                tavily_api_key=settings.TAVILY_API_KEY,
                max_results=3,
                search_depth="basic",
                include_raw_content=False,
                include_answer=False,
                include_domains=include_domains,
                exclude_domains=["facebook.com", "instagram.com", "twitter.com", "youtube.com", "tiktok.com"],
            )
        except Exception as exc:
            logger.error("Could not initialize Tavily search (%s)", type(exc).__name__)
            return _MockSearchTool()

    def get_academic_search_tool(self):
        """Tavily search limited to scholarly sites, so academic search shares web search's API and filtering."""
        return self.get_web_search_tool(include_domains=ACADEMIC_DOMAINS)


def extract_passage(url: str, query: str) -> str | None:
    """Ask Tavily for one query-relevant passage from an already retrieved page."""
    if not settings.TAVILY_API_KEY:
        raise WebSearchUnavailableError("Web extraction is unavailable")
    client = TavilyClient(api_key=settings.TAVILY_API_KEY)
    try:
        response = client.extract(urls=[url], query=query, chunks_per_source=1, timeout=10)
    finally:
        client.session.close()
    if not isinstance(response, dict):
        raise ValueError("Invalid extraction response")
    results = response.get("results")
    if not isinstance(results, list):
        raise ValueError("Invalid extraction response")
    if not results:
        if response.get("failed_results"):
            raise ValueError("Page extraction failed")
        return None
    result = results[0]
    if not isinstance(result, dict) or not isinstance(result.get("url"), str) or result["url"].rstrip("/") != url.rstrip("/"):
        raise ValueError("Extraction response URL does not match the requested page")
    content = result.get("raw_content")
    if not isinstance(content, str):
        raise ValueError("Invalid extracted content")
    return content.strip() or None


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
