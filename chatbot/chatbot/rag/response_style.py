"""Response verbosity: concise by default unless the user asks for detail."""

from __future__ import annotations

import re

# Explicit requests for expanded reasoning / long answers
_EXPAND_HINTS = (
    r"^\s*explain\b",
    r"\bwhy\b",
    r"\bhow (?:does|do|is|are|did|can)\b",
    r"^\s*how\b",
    r"\bshow (?:the )?calculation\b",
    r"\bshow (?:me )?(?:the )?math\b",
    r"\bshow (?:your )?work\b",
    r"\bstep[\s-]by[\s-]step\b",
    r"\bwalk me through\b",
    r"\bgive details\b",
    r"\bin detail\b",
    r"\bdetailed\b",
    r"\btell me more\b",
    r"\belaborate\b",
    r"\bfull explanation\b",
    r"\bbreak(?: it)? down\b",
    r"\bexplain why\b",
    r"\bexplain how\b",
    r"\bexplain (?:in detail|everything|further)\b",
)

_CALC_STEPS_HINTS = (
    r"\bshow (?:the )?calculation\b",
    r"\bshow (?:me )?(?:the )?math\b",
    r"\bshow (?:your )?work\b",
    r"\bstep[\s-]by[\s-]step\b",
    r"\bwalk me through\b",
    r"\bhow (?:did|do) you (?:calculate|get)\b",
    r"\bcalculation steps?\b",
    r"\bhow (?:to |are |is |do i )?calculat",
    r"^\s*explain\s*$",
    r"^\s*why\??\s*$",
)


def wants_expanded_answer(question: str) -> bool:
    """True when the user explicitly wants long reasoning or details."""
    lowered = question.lower()
    return any(re.search(pattern, lowered) for pattern in _EXPAND_HINTS)


def wants_calculation_steps(question: str) -> bool:
    """True only when the user asks to see calculation steps."""
    lowered = question.lower()
    return any(re.search(pattern, lowered) for pattern in _CALC_STEPS_HINTS)


def wants_concise(question: str) -> bool:
    """Default policy: keep answers short unless expansion was requested."""
    return not wants_expanded_answer(question)
