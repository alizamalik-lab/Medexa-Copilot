"""Detect ambiguous billing questions and ask for missing context before the LLM answers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag.memory import ChatMessage
from rag.query_router import CPT_PATTERN, HCPCS_PATTERN

_PAYER_PATTERN = re.compile(
    r"\b(medicare|cms|medicaid|commercial|private payer|workers.?comp|tricare)\b",
    re.IGNORECASE,
)

_VAGUE_REFERENCE = re.compile(
    r"\b(this|that|these|those|it|them|the same|above)\b",
    re.IGNORECASE,
)

_SPECIFIC_FACTUAL = re.compile(
    r"\b(what is|what are|define|describe|explain|tell me about)\b",
    re.IGNORECASE,
)


@dataclass
class BillingContext:
    cpt_codes: set[str] = field(default_factory=set)
    has_payer: bool = False
    has_scenario_detail: bool = False

    @property
    def has_codes(self) -> bool:
        return bool(self.cpt_codes)

    @property
    def has_multiple_codes(self) -> bool:
        return len(self.cpt_codes) >= 2


@dataclass(frozen=True)
class AmbiguousIntent:
    name: str
    patterns: tuple[str, ...]
    needs_codes: bool = False
    needs_multiple_codes: bool = False
    needs_payer: bool = False
    needs_scenario: bool = False


_INTENTS: tuple[AmbiguousIntent, ...] = (
    AmbiguousIntent(
        name="modifier",
        patterns=(
            r"which modifier",
            r"what modifier",
            r"require a modifier",
            r"requires a modifier",
            r"require modifier",
            r"needs a modifier",
            r"need a modifier",
            r"use a modifier",
            r"modifier should i",
            r"modifier 59",
            r"modifier-59",
            r"modifier\s*59",
        ),
        needs_codes=True,
        needs_payer=True,
    ),
    AmbiguousIntent(
        name="bill_together",
        patterns=(
            r"bill these together",
            r"bill them together",
            r"bill together",
            r"same day",
            r"same visit",
            r"on the same date",
        ),
        needs_multiple_codes=True,
    ),
    AmbiguousIntent(
        name="billable",
        patterns=(
            r"can i bill",
            r"is this billable",
            r"is it billable",
            r"bill this",
            r"billable\?",
            r"can this be billed",
        ),
        needs_codes=True,
    ),
    AmbiguousIntent(
        name="compliance",
        patterns=(
            r"is this compliant",
            r"is it compliant",
            r"is this allowed",
            r"is it allowed",
            r"is this covered",
            r"is it covered",
            r"compliant\?",
            r"allowed\?",
            r"covered\?",
        ),
        needs_scenario=True,
    ),
    AmbiguousIntent(
        name="coding_recommendation",
        patterns=(
            r"which cpt should",
            r"what cpt should",
            r"which code should",
            r"what code should i bill",
            r"recommend a cpt",
            r"recommend a code",
            r"what should i bill",
            r"which should i bill",
            r"(?:wound|sq\s*cm).*(?:which|what) cpt",
            r"which cpt.*(?:wound|sq\s*cm|debrid)",
        ),
        needs_scenario=True,
    ),
    AmbiguousIntent(
        name="general_billing",
        patterns=(
            r"phone call",
            r"telephone service",
            r"telephone call",
            r"telehealth",
            r"virtual check[\s-]?in",
            r"can i bill for a \d+[\s-]minute",
        ),
        needs_payer=True,
    ),
)


def _extract_codes(text: str) -> set[str]:
    codes = {match.group(1) for match in CPT_PATTERN.finditer(text)}
    codes.update(match.group(1).upper() for match in HCPCS_PATTERN.finditer(text))
    return codes


def extract_billing_context(
    question: str,
    history: list[ChatMessage] | None = None,
    *,
    include_history: bool = True,
) -> BillingContext:
    parts = [question]
    if history and include_history:
        parts.extend(msg.content for msg in history[-6:])

    combined = "\n".join(parts)
    codes = _extract_codes(combined)
    has_payer = bool(_PAYER_PATTERN.search(combined))

    scenario_chunks = [
        chunk.strip()
        for chunk in parts
        if chunk.strip()
        and not _looks_like_clarification_question(chunk)
        and len(chunk.strip()) >= 40
    ]
    has_scenario = bool(codes) or len(scenario_chunks) >= 1 or len(question.strip()) >= 60

    return BillingContext(
        cpt_codes=codes,
        has_payer=has_payer,
        has_scenario_detail=has_scenario,
    )


def _looks_like_clarification_question(text: str) -> bool:
    lowered = text.lower()
    return "?" in text and any(
        phrase in lowered
        for phrase in (
            "which cpt",
            "could you describe",
            "billing scenario",
            "medicare or",
            "commercial payer",
            "referring to",
        )
    )


def _detect_intent(question: str) -> AmbiguousIntent | None:
    lowered = question.lower().strip()
    if _SPECIFIC_FACTUAL.search(lowered):
        if re.search(r"\b(?:modifier|mod\.?)\s*[#:-]?\s*([A-Z0-9]{2})\b", lowered) or _extract_codes(question):
            return None

    priority = ("coding_recommendation", "general_billing", "modifier", "bill_together", "billable", "compliance")
    for name in priority:
        intent = next((item for item in _INTENTS if item.name == name), None)
        if intent and any(re.search(pattern, lowered) for pattern in intent.patterns):
            return intent

    if not lowered.endswith("?"):
        return None
    if re.search(r"modifier\s*59", lowered) and not _extract_codes(question):
        return next(i for i in _INTENTS if i.name == "modifier")

    for intent in _INTENTS:
        if any(re.search(pattern, lowered) for pattern in intent.patterns):
            return intent
    return None


def _missing_requirements(
    intent: AmbiguousIntent, context: BillingContext, question: str
) -> list[str]:
    missing: list[str] = []

    if intent.needs_codes and not context.has_codes:
        if intent.needs_multiple_codes or re.search(
            r"modifier\s*59", question, re.IGNORECASE
        ):
            missing.append("cpt_codes_multiple")
        else:
            missing.append("cpt_codes")

    if intent.needs_multiple_codes and not context.has_multiple_codes:
        missing.append("cpt_codes_multiple")

    if intent.needs_payer and not context.has_payer:
        if not re.search(r"modifier\s*59", question, re.IGNORECASE):
            missing.append("payer")

    if intent.needs_scenario and not context.has_scenario_detail:
        if _VAGUE_REFERENCE.search(question) or not context.has_codes:
            missing.append("scenario")

    return missing


def _build_clarification(intent: AmbiguousIntent, missing: list[str], question: str) -> str:
    if intent.name == "modifier":
        if re.search(r"modifier\s*59", question, re.IGNORECASE):
            return "Which CPT codes are being billed together?"
        lines = ["Could you tell me:"]
        if "cpt_codes" in missing or "cpt_codes_multiple" in missing:
            lines.append("• Which CPT code(s) are involved?")
        if "payer" in missing:
            lines.append("• Is this Medicare or a commercial payer?")
        return "\n".join(lines)

    if intent.name == "billable":
        return "Which CPT code are you referring to?"

    if intent.name == "bill_together":
        return "Which CPT codes are being billed together on the same date of service?"

    if intent.name == "compliance":
        return (
            "Could you describe the billing scenario or provide the CPT code(s) involved?"
        )

    if intent.name == "coding_recommendation":
        return (
            "I need a little more information before I can recommend the correct billing code.\n\n"
            "• What procedure or service was performed?\n"
            "• What clinical details matter for code selection (site, depth, method, duration)?\n"
            "• Is this Medicare or a commercial payer?"
        )

    if intent.name == "general_billing":
        return (
            "This type of service has its own billing rules, and eligibility depends on "
            "provider type, payer, and documentation.\n\n"
            "Could you tell me:\n"
            "• Are you a PT, OT, or SLP?\n"
            "• Is this Medicare or commercial insurance?"
        )

    return "Could you share the CPT code(s) and payer so I can answer accurately?"


@dataclass(frozen=True)
class ClarificationRequest:
    message: str
    intent_name: str


def detect_clarification_intent(question: str) -> AmbiguousIntent | None:
    return _detect_intent(question)


def _has_sufficient_coding_context(question: str, context: BillingContext) -> bool:
    lowered = question.lower()
    has_procedure_detail = any(
        re.search(pattern, lowered)
        for pattern in (
            r"selective debrid",
            r"non[\s-]?selective debrid",
            r"subcutaneous",
            r"muscle(?:\s+layer)?",
            r"\bbone\b",
            r"full[\s-]?thickness",
            r"partial[\s-]?thickness",
            r"wound depth",
        )
    )
    return context.has_payer and has_procedure_detail


def try_clarification(
    question: str,
    history: list[ChatMessage] | None = None,
    *,
    include_history: bool = True,
) -> ClarificationRequest | None:
    """
    Return a clarification prompt when the question is too ambiguous to answer safely.
    Uses current question plus recent session history to avoid re-asking for known facts.
    """
    # Category G owns phone / online / time_band_select billing (JSON is authoritative).
    from rag.category_tools.time_band import (
        is_category_g_question,
        is_phone_online_question,
        is_time_band_category_question,
    )

    if is_category_g_question(question) or is_time_band_category_question(question):
        return None
    if is_phone_online_question(question):
        return None
    intent = _detect_intent(question)
    if intent is None:
        return None

    context = extract_billing_context(question, history, include_history=include_history)
    missing = _missing_requirements(intent, context, question)
    if intent.name == "coding_recommendation":
        if _has_sufficient_coding_context(question, context):
            return None
        return ClarificationRequest(
            message=_build_clarification(intent, missing or ["scenario"], question),
            intent_name=intent.name,
        )
    if intent.name == "general_billing":
        if context.has_payer:
            return None
        return ClarificationRequest(
            message=_build_clarification(intent, missing or ["payer"], question),
            intent_name=intent.name,
        )
    if not missing:
        return None

    return ClarificationRequest(
        message=_build_clarification(intent, missing, question),
        intent_name=intent.name,
    )
