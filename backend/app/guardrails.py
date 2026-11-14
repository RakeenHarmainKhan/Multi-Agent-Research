"""
Guardrails: cheap, model-free rules.

Input:
- `redact_pii` masks personal data before it reaches the LLM or the search APIs.
- `check_query` flags risky queries. "suspect" means "ask the LLM safety check", "block" means refuse.

Retrieved web content and the report:
- `has_injection` spots pages that try to instruct the model, so they can be dropped.
- `strip_unknown_links` keeps only links to pages we actually retrieved.
"""
import re
from typing import List, Tuple

REFUSAL = "I can't help with that request."

# --- PII ---------------------------------------------------------------------------------------

_EMAIL = re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_SSN = re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")
_CARD = re.compile(r"\b\d(?:[ -]?\d){12,18}\b")
_PHONE = re.compile(r"(?<!\w)(?:\+\d{1,3}[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]\d{3}[ .-]\d{4}(?!\d)")


def _luhn_ok(number: str) -> bool:
    """Card numbers pass the Luhn checksum; random digit strings (ids, dates) mostly don't."""
    digits = [int(c) for c in number if c.isdigit()]
    total = sum(digits[-1::-2]) + sum(sum(divmod(2 * d, 10)) for d in digits[-2::-2])
    return total % 10 == 0


_PII = (
    ("email", _EMAIL, lambda _: True),
    ("ssn", _SSN, lambda _: True),
    ("card", _CARD, _luhn_ok),
    ("phone", _PHONE, lambda _: True),
)


def redact_pii(text: str) -> Tuple[str, List[str]]:
    """Replace emails, SSNs, card numbers and phone numbers with [EMAIL] etc. Returns (text, kinds found)."""
    found: List[str] = []
    for label, pattern, is_valid in _PII:
        def mask(match: re.Match, label: str = label, is_valid=is_valid) -> str:
            if not is_valid(match.group()):
                return match.group()
            if label not in found:
                found.append(label)
            return f"[{label.upper()}]"

        text = pattern.sub(mask, text)
    return text, found


# --- Query rules -------------------------------------------------------------------------------

_BLOCK = re.compile(r"\b(?:make|build|synthesi[sz]e|cook)\s+(?:a\s+|an\s+)?(?:bomb|explosives?|nerve agent|ricin|meth)\b", re.I)

_VIOLENCE_AT_PERSON = re.compile(
    r"\b(?:kill|murder|stab|shoot|strangle|poison|assassinate|hurt|attack|beat up)(?:ing|ed|s)?\s+"
    r"(?:him|her|them|myself|yourself|himself|herself|someone|somebody|people"
    r"|my (?:boss|wife|husband|neighbou?r|coworker|ex|teacher|mother|father))\b",
    re.I,
)

_INJECTION = re.compile(
    r"\b(?:ignore|disregard|forget)\b.{0,30}\b(?:previous|prior|above|earlier|all)\b.{0,20}\b(?:instructions?|rules|prompts?)\b"
    r"|\b(?:reveal|show|print)\b.{0,20}\bsystem prompt\b",
    re.I,
)


def check_query(text: str) -> Tuple[str, str]:
    """Return ("ok" | "suspect" | "block", reason). Rules only; "suspect" is decided by the LLM check."""
    if _BLOCK.search(text):
        return "block", "request for instructions to cause serious harm"
    if _VIOLENCE_AT_PERSON.search(text):
        return "suspect", "violent wording aimed at a person"
    if has_injection(text):
        return "suspect", "prompt-injection wording"
    return "ok", ""


# --- Retrieved content and report output -------------------------------------------------------


def has_injection(text: str) -> bool:
    """True if the text tries to give the model instructions ("ignore previous instructions", ...)."""
    return bool(_INJECTION.search(text))


_MD_LINK = re.compile(r"\[([^\]]*)\]\((https?://[^)\s]+)[^)]*\)")
_BARE_URL = re.compile(r"https?://[^\s)\]>]*[^\s)\]>.,;:!?]")


def _normalize_url(url: str) -> str:
    return url.strip().lower().rstrip("/")


def strip_unknown_links(text: str, allowed_urls: List[str]) -> Tuple[str, int]:
    """
    Keep only links to pages we actually retrieved. A link to any other URL loses its URL
    (markdown links keep their text), so a web page can't plant a link in the report.
    Returns (text, number of links removed).
    """
    allowed = {_normalize_url(url) for url in allowed_urls if url}
    removed = 0

    def drop_markdown_link(match: re.Match) -> str:
        nonlocal removed
        if _normalize_url(match.group(2)) in allowed:
            return match.group(0)
        removed += 1
        return match.group(1)

    def drop_bare_url(match: re.Match) -> str:
        nonlocal removed
        if _normalize_url(match.group(0)) in allowed:
            return match.group(0)
        removed += 1
        return "[link removed]"

    text = _MD_LINK.sub(drop_markdown_link, text)
    return _BARE_URL.sub(drop_bare_url, text), removed
