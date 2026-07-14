"""Route structured billing questions to deterministic tools before RAG."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from rag.memory import ChatMessage
from rag.query_router import CPT_PATTERN, HCPCS_PATTERN, ICD10_PATTERN

ToolName = Literal[
    "lookup_mue",
    "check_ncci",
    "lookup_icd",
    "validate_icd10",
    "lookup_aoc",
    "lookup_cpt",
    "summarize_ncci_restrictions",
    "explain_billing_rules",
]


@dataclass(frozen=True)
class BillingToolIntent:
    tool: ToolName
    params: dict[str, Any]
    reason: str


_MUE_HINTS = (
    r"\bmue\b",
    r"medically unlikely",
    r"unit(?:s)?\s+limit",
    r"max(?:imum)?\s+units",
    r"how many units",
    r"billed\s+\d+\s+units?",
    r"bill(?:ed|ing)?\s+\d+\s+units?",
)

_NCCI_HINTS = (
    r"\bncci\b",
    r"\bptp\b",
    r"ncci restriction",
    r"bill(?:ed)?\s+together",
    r"billed with",
    r"bill(?:ed)?\s+with",
    r"same\s+(?:day|visit|date)",
    r"bundle",
    r"modifier\s*59",
    r"can i bill",
    r"can it be billed",
    r"can they be billed",
    r"allowed together",
)

_ICD_HINTS = (
    r"\bicd[\s-]?10\b",
    r"\bicd\b",
    r"diagnosis code",
    r"valid icd",
    r"icd codes",
)

_ICD_VALIDATION_HINTS = (
    r"valid for",
    r"mapped to",
    r"appropriate for",
    r"support(?:ed)?\s+(?:by|with)",
)

_AOC_HINTS = (
    r"\baoc\b",
    r"add[\s-]?on",
    r"addon",
    r"parent code",
)

_CPT_HINTS = (
    r"\bwhat is\b",
    r"\bwhat are\b",
    r"\bexplain\b",
    r"\bdescribe\b",
    r"\btell me about\b",
    r"list valid",
)

_BILLABLE_HINTS = (
    r"billable",
    r"can i bill",
    r"can this be billed",
    r"can it be billed",
)

_TIMED_HINTS = (
    r"\btimed\b",
    r"time[\s-]based",
    r"8[\s-]minute",
    r"eight[\s-]minute",
)

_RULE_FOLLOWUP_HINTS = (
    r"under which rule",
    r"which rule",
    r"what rule",
    r"how are units calculated",
    r"how is it billed",
    r"billing rule",
)


def _extract_cpt_codes(text: str) -> list[str]:
    codes: list[str] = []
    seen: set[str] = set()
    for match in CPT_PATTERN.finditer(text):
        code = match.group(1)
        if code not in seen:
            seen.add(code)
            codes.append(code)
    for match in HCPCS_PATTERN.finditer(text):
        code = match.group(1).upper()
        if code not in seen:
            seen.add(code)
            codes.append(code)
    return codes


def _extract_icd10_codes(text: str) -> list[str]:
    return [match.group(1).upper() for match in ICD10_PATTERN.finditer(text)]


def _combined_text(question: str, history: list[ChatMessage] | None) -> str:
    parts = [question]
    if history:
        parts.extend(msg.content for msg in history[-8:])
    return "\n".join(parts)


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def _intent_key(intent: BillingToolIntent) -> tuple[str, tuple[tuple[str, Any], ...]]:
    return intent.tool, tuple(sorted(intent.params.items()))


def _detect_icd_validation(
    question: str, cpt_codes: list[str], icd_codes: list[str]
) -> BillingToolIntent | None:
    if not cpt_codes or not icd_codes:
        return None
    if not (
        _matches_any(question, _ICD_VALIDATION_HINTS)
        or re.search(r"\bvalid\b", question, re.IGNORECASE)
    ):
        return None
    return BillingToolIntent(
        tool="validate_icd10",
        params={"cpt_code": cpt_codes[0], "icd10_code": icd_codes[0]},
        reason="icd_validation",
    )


def is_multi_topic_question(question: str) -> bool:
    lowered = question.lower()
    if question.count("?") >= 2:
        return True
    if len(question) >= 180:
        return True

    topic_markers = (
        "explain",
        "what is",
        "what are",
        "timed",
        "mue",
        "icd",
        "ncci",
        "modifier",
        "bill",
        "medicare",
        "ama",
        "units",
        "calculate",
        "together",
        "add-on",
        "add on",
        "addon",
        "aoc",
    )
    hits = sum(1 for marker in topic_markers if marker in lowered)
    if hits >= 2 and (" and " in lowered or "," in lowered):
        return True
    return hits >= 3


def detect_all_billing_tool_intents(
    question: str,
    history: list[ChatMessage] | None = None,
    focus_code: str | None = None,
    *,
    use_focus_code: bool = True,
) -> list[BillingToolIntent]:
    lowered = question.lower().strip()
    combined = _combined_text(question, history) if use_focus_code else question
    codes = _extract_cpt_codes(combined if use_focus_code else question)
    icd_codes = _extract_icd10_codes(question)
    if not codes and focus_code and use_focus_code:
        codes = [focus_code]

    intents: list[BillingToolIntent] = []

    icd_validation = _detect_icd_validation(question, codes, icd_codes)
    if icd_validation:
        intents.append(icd_validation)
        deduped: dict[tuple[str, tuple[tuple[str, Any], ...]], BillingToolIntent] = {
            _intent_key(icd_validation): icd_validation
        }
        return list(deduped.values())

    if codes and _matches_any(lowered, _RULE_FOLLOWUP_HINTS):
        intents.append(
            BillingToolIntent(
                tool="explain_billing_rules",
                params={"cpt_code": codes[0]},
                reason="billing_rule_followup",
            )
        )

    if codes and (
        _matches_any(lowered, _MUE_HINTS) or re.search(r"\bits?\s+mue\b", lowered)
    ):
        for code in codes:
            intents.append(
                BillingToolIntent(
                    tool="lookup_mue",
                    params={"cpt_code": code},
                    reason="mue_lookup",
                )
            )

    if len(codes) >= 2 and _matches_any(lowered, _NCCI_HINTS):
        # Check every unique pair — never stop after the first two codes.
        for i, left in enumerate(codes):
            for right in codes[i + 1 :]:
                intents.append(
                    BillingToolIntent(
                        tool="check_ncci",
                        params={"cpt1": left, "cpt2": right},
                        reason="ncci_pair_lookup",
                    )
                )
    elif len(codes) == 1 and _matches_any(lowered, _NCCI_HINTS):
        intents.append(
            BillingToolIntent(
                tool="summarize_ncci_restrictions",
                params={"cpt_code": codes[0]},
                reason="ncci_summary_lookup",
            )
        )

    if codes and _matches_any(lowered, _ICD_HINTS) and not icd_codes:
        for code in codes:
            intents.append(
                BillingToolIntent(
                    tool="lookup_icd",
                    params={"cpt_code": code},
                    reason="icd_lookup",
                )
            )

    if codes and _matches_any(lowered, _AOC_HINTS):
        for code in codes:
            intents.append(
                BillingToolIntent(
                    tool="lookup_aoc",
                    params={"cpt_code": code},
                    reason="aoc_lookup",
                )
            )

    if codes and (
        _matches_any(lowered, _CPT_HINTS)
        or _matches_any(lowered, _BILLABLE_HINTS)
        or _matches_any(lowered, _TIMED_HINTS)
    ):
        for code in codes:
            intents.append(
                BillingToolIntent(
                    tool="lookup_cpt",
                    params={"cpt_code": code},
                    reason="cpt_lookup",
                )
            )
    elif codes and len(codes) == 1 and re.search(
        r"\b(is|are)\s+(?:it|this|that)\b", lowered
    ):
        intents.append(
            BillingToolIntent(
                tool="lookup_cpt",
                params={"cpt_code": codes[0]},
                reason="cpt_lookup",
            )
        )

    deduped = {}
    for intent in intents:
        deduped[_intent_key(intent)] = intent
    return list(deduped.values())


def detect_billing_tool_intent(
    question: str,
    history: list[ChatMessage] | None = None,
    focus_code: str | None = None,
) -> BillingToolIntent | None:
    intents = detect_all_billing_tool_intents(question, history, focus_code)
    if not intents:
        return None
    return intents[0]
