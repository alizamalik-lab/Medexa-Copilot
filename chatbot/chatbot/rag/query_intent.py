"""High-level user intent classification for routing and logging."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from rag.conversation_context import extract_cpt_codes_ordered
from rag.billing_engine import (
    is_unit_calculation_guide_question,
    is_unit_calculation_question,
)


class UserIntent(str, Enum):
    CPT_EXPLANATION = "cpt_explanation"
    MUE_LOOKUP = "mue_lookup"
    NCCI_LOOKUP = "ncci_lookup"
    ICD_VALIDATION = "icd_validation"
    MODIFIER_RECOMMENDATION = "modifier_recommendation"
    BILLING_UNIT_CALCULATION = "billing_unit_calculation"
    CODING_RECOMMENDATION = "coding_recommendation"
    DOCUMENTATION_GUIDANCE = "documentation_guidance"
    GENERAL_BILLING = "general_billing"
    GENERAL_HEALTHCARE_KNOWLEDGE = "general_healthcare_knowledge"
    OUT_OF_SCOPE = "out_of_scope"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ClassifiedIntent:
    primary: UserIntent
    secondary: tuple[UserIntent, ...] = ()


_CPT_EXPLAIN = (
    r"\bwhat is\b",
    r"\bwhat are\b",
    r"\bexplain\b",
    r"\bdescribe\b",
    r"\btell me about\b",
)
_MUE = (r"\bmue\b", r"medically unlikely", r"unit(?:s)?\s+limit")
_NCCI = (r"\bncci\b", r"\bptp\b", r"bill(?:ed)?\s+together", r"billed with")
_ICD = (r"\bicd[\s-]?10\b", r"valid icd", r"diagnosis code")
_MODIFIER = (r"which modifier", r"what modifier", r"modifier\s*59")
_CODING_REC = (
    r"which cpt should",
    r"what cpt should",
    r"which code should",
    r"what code should i bill",
    r"recommend a cpt",
    r"recommend a code",
)
_DOCUMENTATION = (
    r"documentation",
    r"document",
    r"what should i document",
)
_GENERAL_BILLING = (
    r"phone call",
    r"telephone",
    r"telehealth",
    r"virtual check",
    r"can i bill for a \d+",
)
_HEALTHCARE = (
    r"hipaa",
    r"claim denial",
    r"reimbursement",
    r"payer policy",
)


def classify_user_intent(question: str) -> ClassifiedIntent:
    lowered = question.lower().strip()
    codes = extract_cpt_codes_ordered(question)
    intents: list[UserIntent] = []

    if is_unit_calculation_question(question) or is_unit_calculation_guide_question(question):
        intents.append(UserIntent.BILLING_UNIT_CALCULATION)
    if any(re.search(p, lowered) for p in _CODING_REC):
        intents.append(UserIntent.CODING_RECOMMENDATION)
    if any(re.search(p, lowered) for p in _GENERAL_BILLING) and not codes:
        intents.append(UserIntent.GENERAL_BILLING)
    if any(re.search(p, lowered) for p in _MUE):
        intents.append(UserIntent.MUE_LOOKUP)
    if any(re.search(p, lowered) for p in _NCCI):
        intents.append(UserIntent.NCCI_LOOKUP)
    if any(re.search(p, lowered) for p in _ICD):
        intents.append(UserIntent.ICD_VALIDATION)
    if any(re.search(p, lowered) for p in _MODIFIER):
        intents.append(UserIntent.MODIFIER_RECOMMENDATION)
    if codes and any(re.search(p, lowered) for p in _CPT_EXPLAIN):
        intents.append(UserIntent.CPT_EXPLANATION)
    if any(re.search(p, lowered) for p in _DOCUMENTATION):
        intents.append(UserIntent.DOCUMENTATION_GUIDANCE)
    if any(re.search(p, lowered) for p in _HEALTHCARE):
        intents.append(UserIntent.GENERAL_HEALTHCARE_KNOWLEDGE)

    if not intents:
        if codes:
            intents.append(UserIntent.CPT_EXPLANATION)
        else:
            intents.append(UserIntent.GENERAL_HEALTHCARE_KNOWLEDGE)

    primary = intents[0]
    secondary = tuple(intent for intent in intents[1:] if intent != primary)
    return ClassifiedIntent(primary=primary, secondary=secondary)
