"""Detect out-of-scope questions and return polite redirects before any LLM/RAG call."""

from __future__ import annotations

import re

SCOPE_REDIRECT_GENERAL = (
    "I'm Medexa, a Medical Billing and Healthcare Copilot. I can assist with "
    "medical billing, coding, compliance, healthcare documentation, and "
    "PT/OT/SLP-related questions. I can't answer questions outside these areas."
)

SCOPE_REDIRECT_PROGRAMMING = (
    "I'm Medexa, a Medical Billing and Healthcare Copilot. I specialize in "
    "medical billing, coding, compliance, and healthcare documentation. "
    "I can't assist with programming questions, but I'd be happy to help with "
    "any healthcare billing or coding questions."
)

_IN_SCOPE_HINTS = (
    r"\bcpt\b",
    r"\bicd",
    r"\bhcpcs\b",
    r"\bmedicare\b",
    r"\bmedicaid\b",
    r"\bbilling\b",
    r"\bmodifier\b",
    r"\bncci\b",
    r"\bmue\b",
    r"\bhipaa\b",
    r"\bclaim\b",
    r"\btherapy\b",
    r"\bphysical therapy\b",
    r"\boccupational therapy\b",
    r"\bspeech[\s-]?language\b",
    r"\bdocumentation\b",
    r"\bcompliance\b",
    r"\breimbursement\b",
    r"\bpayer\b",
    r"\b8[\s-]minute\b",
    r"\brule of eight\b",
    r"\bevaluation\b",
    r"\bre-?evaluation\b",
    r"\b97110\b",
    r"\b97530\b",
    r"\bmedexa\b",
)

_OUT_OF_SCOPE_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\b(python|javascript|java\b|typescript|react|node\.?js|write a program|"
            r"coding tutorial|software develop|debug my code|html|css)\b",
            re.IGNORECASE,
        ),
        SCOPE_REDIRECT_PROGRAMMING,
    ),
    (
        re.compile(
            r"\b(fifa|world cup|super bowl|nba|nfl|mlb|soccer|football match|"
            r"who won the|election|president|politics|movie|netflix|recipe|cook|"
            r"celebrity|tiktok|instagram)\b",
            re.IGNORECASE,
        ),
        SCOPE_REDIRECT_GENERAL,
    ),
)


def is_in_scope(question: str) -> bool:
    lowered = question.lower()
    return any(re.search(pattern, lowered) for pattern in _IN_SCOPE_HINTS)


def try_scope_redirect(question: str) -> str | None:
    """
    Return a redirect message for clearly out-of-scope questions.
    Returns None when the question appears to be within Medexa's domain.
    """
    if is_in_scope(question):
        return None

    for pattern, message in _OUT_OF_SCOPE_RULES:
        if pattern.search(question):
            return message

    return None
