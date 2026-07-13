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
    resolved_focus = focus_code or infer_focus_code(history, focus_code)
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

    return ResolvedQuestion(text=enriched, focus_code=resolved_focus)


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
    if enriched.focus_code is None and stored_focus:
        enriched = ResolvedQuestion(text=enriched.text, focus_code=stored_focus)
    return enriched


def update_focus_code(
    previous_focus: str | None,
    question: str,
    effective_question: str,
    history: list[ChatMessage],
) -> str | None:
    for text in (effective_question, question):
        codes = extract_cpt_codes_ordered(text)
        if codes:
            return codes[0]

    context = extract_billing_context(effective_question, history)
    if context.cpt_codes:
        return sorted(context.cpt_codes)[0]

    return previous_focus


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
    return bool(_extract_payer_phrase(question))
