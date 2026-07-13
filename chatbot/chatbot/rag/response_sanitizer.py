"""Remove implementation language from user-facing assistant responses."""

from __future__ import annotations

import re

_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\bthe tool result(?: doesn't| does not| didn't| did not| don't| do not| hasn't| has not)?[^.?!]*[.?!]?",
            re.IGNORECASE,
        ),
        "I couldn't confirm that from the available billing data.",
    ),
    (
        re.compile(
            r"\bthe (?:tool|lookup|retrieved|database) (?:result|returned|response|data)[^.?!]*[.?!]?",
            re.IGNORECASE,
        ),
        "Based on the available billing information,",
    ),
    (
        re.compile(r"\bthe retrieved context[^.?!]*[.?!]?", re.IGNORECASE),
        "Based on the available billing information,",
    ),
    (
        re.compile(r"\bthe json(?: data)?[^.?!]*[.?!]?", re.IGNORECASE),
        "the billing information I have available",
    ),
    (
        re.compile(r"\baccording to the (?:tool|lookup|retrieval|vector|embedding)[^.?!]*[.?!]?", re.IGNORECASE),
        "Based on the available billing information,",
    ),
    (
        re.compile(r"\bthe tool suggests\b", re.IGNORECASE),
        "Based on the billing rules available in Medexa,",
    ),
    (
        re.compile(r"\bdivide.{0,20}by\s*8\b", re.IGNORECASE),
        "use the CMS 8-Minute Rule conversion table",
    ),
    (
        re.compile(r"\bround to the nearest whole number\b", re.IGNORECASE),
        "apply the CMS conversion table for timed therapy minutes",
    ),
    (
        re.compile(r"\bi couldn't confirm that\b[^.?!]*[.?!]?", re.IGNORECASE),
        "",
    ),
    (
        re.compile(r"\bi don't have enough information\b[^.?!]*[.?!]?", re.IGNORECASE),
        "",
    ),
    (
        re.compile(r"\bsample codes?\b", re.IGNORECASE),
        "examples",
    ),
)

_FORBIDDEN_PHRASES = (
    "tool result",
    "lookup returned",
    "retrieved context",
    "vector database",
    "embedding",
    "json doesn't",
    "json does not",
    "the database",
)


def sanitize_response(answer: str) -> str:
    cleaned = answer.strip()
    for pattern, replacement in _REPLACEMENTS:
        cleaned = pattern.sub(replacement, cleaned)

    lowered = cleaned.lower()
    if any(phrase in lowered for phrase in _FORBIDDEN_PHRASES):
        cleaned = re.sub(
            r"\b(?:tool|lookup|retrieval|vector|embedding|json|database|context)\b",
            "billing information",
            cleaned,
            flags=re.IGNORECASE,
        )

    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()
