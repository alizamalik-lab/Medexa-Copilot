"""Expand the previous answer when the user asks Explain / Why / How / etc."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag.billing_categories import get_billing_category_store
from rag.billing_engine import (
    EIGHT_MINUTE_RULE,
    UNTIMED_RULES,
    calculate_ama_units,
    calculate_cms_pooled_units,
    eight_minute_units,
    is_cms_ama_comparison_question,
    is_rule_comparison_question,
    is_unit_calculation_question,
    parse_cpt_time_entries,
    _extract_billing_codes,
    _has_implicit_time_documentation,
)
from rag.billing_orchestrator import explain_special_category
from rag.memory import ChatMessage


@dataclass(frozen=True)
class FollowupExplanation:
    answer: str
    sources: list[str]


_EXPLAIN_FOLLOWUP = re.compile(
    r"^\s*("
    r"explain(?:\s+please)?|"
    r"why(?:\s+(?:is\s+that|that|so))?|"
    r"how(?:\s+(?:so|come))?|"
    r"show(?:\s+me)?(?:\s+the)?(?:\s+calculation|\s+math|\s+work)?|"
    r"show\s+me\s+the\s+math|"
    r"can you elaborate(?:\s+please)?|"
    r"elaborate(?:\s+please)?|"
    r"tell me more|"
    r"more detail(?:s)?|"
    r"break(?:\s+it)?\s+down|"
    r"walk me through(?:\s+it)?"
    r")\s*[?.!]?\s*$",
    re.IGNORECASE,
)


def is_explain_followup(question: str) -> bool:
    """True for short follow-ups that mean 'explain the previous answer'."""
    return bool(_EXPLAIN_FOLLOWUP.match(question.strip()))


def _last_user_and_assistant(
    history: list[ChatMessage],
) -> tuple[str | None, str | None]:
    last_user = None
    last_assistant = None
    for msg in reversed(history):
        if last_assistant is None and msg.role == "assistant":
            last_assistant = msg.content
        elif last_user is None and msg.role == "user":
            last_user = msg.content
        if last_user and last_assistant:
            break
    return last_user, last_assistant


def try_followup_explanation(
    question: str,
    history: list[ChatMessage],
) -> FollowupExplanation | None:
    """
    If the user asks to explain the prior answer, expand it using known context.
    Does not re-ask for CPT/payer already present in the previous turn.
    """
    if not is_explain_followup(question):
        return None
    if not history:
        return FollowupExplanation(
            answer=(
                "What would you like me to explain? "
                "Share the CPT code or paste the billing question."
            ),
            sources=[],
        )

    last_user, last_assistant = _last_user_and_assistant(history)
    if not last_user:
        return FollowupExplanation(
            answer=(
                "I'm not sure which answer to expand. "
                "Ask again with the CPT code or billing scenario."
            ),
            sources=[],
        )

    # CMS vs AMA comparison expansion
    if (
        is_cms_ama_comparison_question(last_user)
        or is_rule_comparison_question(last_user)
        or (
            last_assistant
            and "Medicare (CMS)" in last_assistant
            and "AMA" in last_assistant
        )
    ):
        expanded = _explain_cms_ama(last_user)
        if expanded:
            return FollowupExplanation(
                answer=expanded,
                sources=["billing_engine"],
            )

    # Unit calculation expansion
    if (
        is_unit_calculation_question(last_user)
        or _has_implicit_time_documentation(last_user)
        or (last_assistant and re.search(r"(Medicare )?Units?:\s*\*?\*?\d+", last_assistant or ""))
    ):
        expanded = _explain_unit_calculation(last_user, last_assistant)
        if expanded:
            return FollowupExplanation(
                answer=expanded,
                sources=["billing_engine", "billing_category_engine"],
            )

    # CPT / category / MUE-style prior answers
    codes = _extract_billing_codes(last_user)
    if not codes and last_assistant:
        codes = _extract_billing_codes(last_assistant)
    if codes:
        expanded = _explain_code_context(codes[0], last_user, last_assistant)
        if expanded:
            return FollowupExplanation(
                answer=expanded,
                sources=["billing_category_engine"],
            )

    # Generic short expansion of previous assistant text
    if last_assistant:
        return FollowupExplanation(
            answer=_shorten_prior_as_explanation(last_assistant),
            sources=[],
        )

    return FollowupExplanation(
        answer=(
            "I can expand the previous billing answer if you restate the question "
            "or CPT code."
        ),
        sources=[],
    )


def _explain_cms_ama(prior_question: str) -> str | None:
    entries = parse_cpt_time_entries(prior_question)
    if not entries:
        return (
            "**Medicare (CMS)** pools timed therapy minutes across eligible codes, "
            "then converts the total with the 8-minute table.\n\n"
            "**AMA Rule of Eight** calculates each CPT independently; leftover "
            "minutes on one code cannot create units for another.\n\n"
            "**Why they differ:** CMS combines time first; AMA does not."
        )

    cms = calculate_cms_pooled_units(entries)
    ama = calculate_ama_units(entries)
    cms_total = sum(cms.values())
    ama_total = sum(ama.values())
    total_minutes = sum(e.minutes for e in entries)
    minute_parts = " + ".join(str(e.minutes) for e in entries)

    lines = [
        "**Medicare (CMS)**",
        f"• Pool all timed minutes: {minute_parts} = **{total_minutes}** minutes.",
        f"• {total_minutes} pooled minutes = **{cms_total}** billable unit"
        f"{'' if cms_total == 1 else 's'}.",
        "",
        "**AMA Rule of Eight**",
        "• Evaluate each CPT separately.",
    ]
    for entry in entries:
        unit = ama.get(entry.cpt_code, eight_minute_units(entry.minutes))
        lines.append(
            f"• {entry.cpt_code}: {entry.minutes} minutes = **{unit}** unit"
            f"{'' if unit == 1 else 's'}."
        )
    lines.append(
        f"• Total = **{ama_total}** unit{'' if ama_total == 1 else 's'}."
    )
    lines.extend(
        [
            "",
            "**Why they're different**",
            "CMS combines the treatment time before calculating units, while AMA "
            "calculates each CPT code independently.",
        ]
    )
    return "\n".join(lines)


def _explain_unit_calculation(
    prior_question: str, prior_answer: str | None
) -> str | None:
    entries = parse_cpt_time_entries(prior_question)
    if not entries:
        return None

    store = get_billing_category_store()
    lines: list[str] = []
    for entry in entries:
        profile = store.get_profile(entry.cpt_code)
        rule = profile.billing_rule if profile else "unknown"
        if rule == EIGHT_MINUTE_RULE:
            units = eight_minute_units(entry.minutes)
            lines.append(f"**{entry.cpt_code}** (CMS 8-Minute Rule)")
            lines.append(f"• Documented time: **{entry.minutes}** minutes.")
            lines.append(
                f"• Conversion: {entry.minutes} minutes → **{units}** unit"
                f"{'' if units == 1 else 's'}."
            )
        elif rule in UNTIMED_RULES:
            lines.append(f"**{entry.cpt_code}** (untimed)")
            lines.append(
                f"• Time does not drive units; bill **1** "
                f"({profile.max_units_allowed if profile else 'per category'})."
            )
        else:
            cat = explain_special_category(entry.cpt_code, concise=True)
            lines.append(f"**{entry.cpt_code}**")
            lines.append(f"• Documented time: **{entry.minutes}** minutes.")
            if cat:
                lines.append(f"• {cat}")
        lines.append("")

    if len(entries) > 1:
        all_eight = True
        for entry in entries:
            profile = store.get_profile(entry.cpt_code)
            if profile is None or profile.billing_rule != EIGHT_MINUTE_RULE:
                all_eight = False
                break
        if all_eight:
            pooled = sum(calculate_cms_pooled_units(entries).values())
            total = sum(e.minutes for e in entries)
            lines.append(
                f"**CMS pooled total:** {total} minutes → **{pooled}** unit"
                f"{'' if pooled == 1 else 's'}."
            )

    text = "\n".join(line for line in lines if line is not None).strip()
    return text[:900] if text else prior_answer


def _explain_code_context(
    code: str, prior_question: str, prior_answer: str | None
) -> str:
    store = get_billing_category_store()
    profile = store.get_profile(code)
    parts: list[str] = [f"**CPT {code}**"]

    if profile:
        parts.append(
            f"• Billing rule: `{profile.billing_rule}` "
            f"(Category {profile.category_id}"
            + (f"/{profile.subcategory_id}" if profile.subcategory_id else "")
            + ")."
        )
        parts.append(f"• {store.rule_definition(profile.billing_rule)}")
        if profile.max_units_allowed:
            parts.append(f"• Max units: {profile.max_units_allowed}.")
        if profile.min_time_for_1_unit:
            parts.append(f"• Minimum requirement: {profile.min_time_for_1_unit}.")
    elif prior_answer:
        # Fall back to a tightened rewrite of the prior answer
        return _shorten_prior_as_explanation(prior_answer)

    # Keep 5–8 lines
    return "\n".join(parts[:8])


def _shorten_prior_as_explanation(prior_answer: str) -> str:
    lines = [ln.strip() for ln in prior_answer.splitlines() if ln.strip()]
    # Prefer bullet-ish content
    kept = lines[:8]
    if not kept:
        return prior_answer[:500]
    header = "Here's a bit more detail on that answer:"
    return header + "\n\n" + "\n".join(kept)
