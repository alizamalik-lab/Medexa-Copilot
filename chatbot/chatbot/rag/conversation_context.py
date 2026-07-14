"""Resolve pronouns and continue pending clarification turns."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag.clarification import extract_billing_context
from rag.memory import ChatMessage, PendingClarification
from rag.query_router import CPT_PATTERN, HCPCS_PATTERN

_PRONOUN_REFERENCE = re.compile(
    r"\b(it|its|it's|this|that|these|those|the same|the code|the cpt)\b",
    re.IGNORECASE,
)

_SHORT_CODE_REPLY = re.compile(r"^\s*(\d{5}|[A-VJ-Z]\d{4})(?:\s+and\s+(\d{5}|[A-VJ-Z]\d{4}))?\s*$", re.IGNORECASE)

_CPT_IN_TEXT = re.compile(
    r"\b(?:cpt|hcpcs|code)\s*[#:]?\s*(\d{5}|[A-VJ-Z]\d{4})\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ResolvedQuestion:
    text: str
    merged_from_pending: bool = False
    focus_code: str | None = None
    reset_focus: bool = False


_NEW_TOPIC_PATTERNS = (
    r"which cpt (?:code )?should",
    r"what cpt (?:code )?should",
    r"which code should",
    r"what code should",
    r"recommend (?:a )?(?:cpt|code)",
    r"\bphone call\b",
    r"\btelephone\b",
    r"\btelehealth\b",
    r"can i bill for a \d+",
    r"\d+[\s-]minute (?:phone|call|telephone)",
    r"which cpt.*(?:wound|sq\s*cm|debrid)",
    r"(?:wound|sq\s*cm|debrid).*(?:which|what) cpt",
)

_FOLLOWUP_PATTERNS = (
    r"under which rule",
    r"which rule",
    r"what rule",
    r"how are units calculated",
    r"how is it billed",
    r"\bis it timed\b",
    r"\bis it billable\b",
    r"\bits?\s+mue\b",
    r"\bits?\s+ncci\b",
    r"\bthe same code\b",
    r"^\s*explain\b",
    r"^\s*why\b",
    r"^\s*how\b",
    r"^\s*show\b",
    r"^\s*(?:name|list|names)\b",
    r"\bname them\b",
    r"\blist them\b",
    r"tell me more",
    r"elaborate",
    r"show me the math",
    r"show calculation",
)


def extract_cpt_codes_ordered(text: str) -> list[str]:
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


def references_prior_topic(question: str) -> bool:
    if _PRONOUN_REFERENCE.search(question):
        return True
    lowered = question.lower()
    return any(re.search(pattern, lowered) for pattern in _FOLLOWUP_PATTERNS)


def is_independent_topic(question: str, stored_focus: str | None) -> bool:
    if not stored_focus:
        return False

    codes = extract_cpt_codes_ordered(question)
    if codes and codes[0] != stored_focus:
        return True

    if references_prior_topic(question):
        return False

    lowered = question.lower()
    return any(re.search(pattern, lowered) for pattern in _NEW_TOPIC_PATTERNS)


def infer_focus_code(history: list[ChatMessage], stored_focus: str | None = None) -> str | None:
    if stored_focus:
        return stored_focus

    for message in reversed(history):
        codes = extract_cpt_codes_ordered(message.content)
        if codes:
            return codes[0]

        label_match = _CPT_IN_TEXT.search(message.content)
        if label_match:
            return label_match.group(1).upper()
    return None


def merge_pending_clarification(
    question: str,
    history: list[ChatMessage],
    pending: PendingClarification | None,
) -> ResolvedQuestion:
    if pending is None:
        return ResolvedQuestion(text=question)

    # New independent billing turn — do not merge into the pending prompt.
    if _looks_like_new_billing_turn(question):
        return ResolvedQuestion(text=question)

    reply_codes = extract_cpt_codes_ordered(question)
    short_reply = _SHORT_CODE_REPLY.match(question.strip())
    if short_reply:
        first = short_reply.group(1).upper()
        second = short_reply.group(2)
        reply_codes = [first] + ([second.upper()] if second else [])

    if not reply_codes and not _looks_like_clarification_answer(question, pending):
        return ResolvedQuestion(text=question)

    original = pending.original_question.lower()
    code_text = " and ".join(reply_codes)
    payer = _extract_payer_phrase(question)

    if pending.intent_name == "billable":
        merged = f"Is CPT {code_text} billable?"
    elif pending.intent_name == "modifier":
        merged = f"Which modifier should I use for CPT {code_text}"
        if payer:
            merged += f" on {payer}"
        merged += "?"
    elif pending.intent_name == "bill_together":
        merged = f"Can {code_text} be billed together?"
    elif pending.intent_name == "compliance":
        merged = (
            f"Is billing CPT {code_text} compliant"
            + (f" for {question.strip()}" if not reply_codes else "")
            + "?"
        )
    elif pending.intent_name == "coding_recommendation":
        merged = f"{pending.original_question} {question.strip()}".strip()
    elif pending.intent_name == "general_billing":
        merged = f"{pending.original_question} ({question.strip()})"
    elif pending.intent_name == "category_clarification":
        merged = f"{pending.original_question} {question.strip()}".strip()
    else:
        merged = f"{pending.original_question} ({question})"

    focus = reply_codes[0] if reply_codes else None
    return ResolvedQuestion(
        text=merged,
        merged_from_pending=True,
        focus_code=focus,
    )


def enrich_question_with_context(
    question: str,
    history: list[ChatMessage],
    focus_code: str | None,
) -> ResolvedQuestion:
    if focus_code and is_independent_topic(question, focus_code):
        codes_in_question = extract_cpt_codes_ordered(question)
        return ResolvedQuestion(
            text=question,
            focus_code=codes_in_question[0] if codes_in_question else None,
            reset_focus=True,
        )

    resolved_focus = focus_code if references_prior_topic(question) else None
    if not resolved_focus:
        resolved_focus = infer_focus_code(history) if references_prior_topic(question) else None

    enriched = _enrich_contextual_followup(question, resolved_focus)

    if _PRONOUN_REFERENCE.search(enriched) and resolved_focus:
        enriched = _inject_focus_code(enriched, resolved_focus)
        codes = extract_cpt_codes_ordered(enriched)
        return ResolvedQuestion(
            text=enriched,
            focus_code=codes[0] if codes else resolved_focus,
        )

    codes_in_question = extract_cpt_codes_ordered(enriched)
    if codes_in_question:
        return ResolvedQuestion(text=enriched, focus_code=codes_in_question[0])

    if references_prior_topic(question) and resolved_focus:
        return ResolvedQuestion(text=enriched, focus_code=resolved_focus)

    return ResolvedQuestion(text=enriched, focus_code=None)


def _enrich_contextual_followup(question: str, focus_code: str | None) -> str:
    if not focus_code:
        return question
    lowered = question.lower().strip()
    followup_patterns = (
        r"under which rule",
        r"which rule",
        r"what rule",
        r"how are units calculated",
        r"how is it billed",
    )
    if any(re.search(pattern, lowered) for pattern in followup_patterns):
        return f"What billing rule applies to CPT {focus_code}?"
    return question


def resolve_effective_question(
    question: str,
    history: list[ChatMessage],
    pending: PendingClarification | None,
    stored_focus: str | None,
) -> ResolvedQuestion:
    merged = merge_pending_clarification(question, history, pending)
    if merged.merged_from_pending:
        return merged

    enriched = enrich_question_with_context(merged.text, history, stored_focus)
    return enriched


def resolve_active_focus(
    question: str,
    resolved: ResolvedQuestion,
    stored_focus: str | None,
) -> str | None:
    if resolved.reset_focus:
        return resolved.focus_code
    if resolved.focus_code:
        return resolved.focus_code
    if references_prior_topic(question):
        return stored_focus
    return None


def update_focus_code(
    previous_focus: str | None,
    question: str,
    effective_question: str,
    history: list[ChatMessage],
    *,
    include_history: bool = True,
) -> str | None:
    for text in (effective_question, question):
        codes = extract_cpt_codes_ordered(text)
        if codes:
            return codes[0]

    if is_independent_topic(question, previous_focus):
        return None

    if include_history:
        context = extract_billing_context(
            effective_question, history, include_history=True
        )
        if context.cpt_codes:
            return sorted(context.cpt_codes)[0]

    if references_prior_topic(question):
        return previous_focus

    return None


def _inject_focus_code(question: str, focus_code: str) -> str:
    lowered = question.lower().strip()
    if focus_code in question:
        return question

    replacements = (
        (r"\bits\b", f"CPT {focus_code}'s"),
        (r"\bit\b", f"CPT {focus_code}"),
        (r"\bthis\b", f"CPT {focus_code}"),
        (r"\bthat\b", f"CPT {focus_code}"),
        (r"\bthe code\b", f"CPT {focus_code}"),
        (r"\bthe cpt\b", f"CPT {focus_code}"),
    )
    enriched = question
    for pattern, replacement in replacements:
        enriched = re.sub(pattern, replacement, enriched, count=1, flags=re.IGNORECASE)

    if focus_code not in enriched:
        enriched = f"{enriched.strip()} (regarding CPT {focus_code})"
    return enriched


def _extract_payer_phrase(text: str) -> str | None:
    match = re.search(
        r"\b(on\s+)?(medicare|cms|medicaid|commercial(?:\s+payer)?|tricare)\b",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    return match.group(2)


def _looks_like_clarification_answer(
    question: str, pending: PendingClarification
) -> bool:
    if pending.intent_name == "compliance" and len(question.strip()) >= 20:
        return True
    if pending.intent_name == "category_clarification":
        lowered = question.lower().strip()
        # Reject clearly new billing scenarios (do not swallow the next turn).
        if _looks_like_new_billing_turn(question):
            return False
        option_phrases = (
            "selective",
            "debrid",
            "npwt",
            "negative pressure",
            "wound vac",
            "disposable",
            "traditional",
            "telephone assessment",
            "online digital",
            "brief communication",
            "check-in",
            "check in",
            "portal",
        )
        if any(phrase in lowered for phrase in option_phrases):
            return True
        # Allow very short explicit picks like "Telephone" / "Selective debridement"
        if len(lowered) <= 60 and any(
            phrase in lowered
            for phrase in (
                "telephone",
                "phone",
                "online",
                "digital",
                "selective",
                "traditional",
                "disposable",
            )
        ):
            return True
        return False
    return bool(_extract_payer_phrase(question))


def _looks_like_new_billing_turn(question: str) -> bool:
    """True when the user started a new billing question instead of answering a prompt."""
    lowered = question.lower().strip()
    if "?" in question and len(lowered) > 40:
        return True
    if re.search(
        r"\b(i performed|i treated|i billed|how many units|compare medicare|"
        r"what is the mue|can (?:cpt|i) bill|spoke for \d+|called me for)\b",
        lowered,
    ):
        return True
    if CPT_PATTERN.search(question) and re.search(
        r"\b(\d{1,3}\s*min|units?|mue|ncci|timed|bill)\b", lowered
    ):
        return True
    return False
