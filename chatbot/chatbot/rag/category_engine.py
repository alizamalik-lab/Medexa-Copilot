"""Deterministic billing category engine — primary rules route before the LLM.

Decision flow:
  Intent → Extract CPT/keywords → Find category → Apply category tool
  → (optional) MUE / NCCI structure → Structured result → LLM explains

The LLM must never invent billing rules, unit math, or max-unit caps.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from rag.billing_categories import get_billing_category_store
from rag.billing_engine import (
    AREA_BASED,
    TIME_BAND_SELECT,
    detect_rule_methodology,
    is_unit_calculation_guide_question,
    is_unit_calculation_question,
    parse_area_sq_cm,
    parse_cpt_time_entries,
    _extract_billing_codes,
    _has_implicit_time_documentation,
)
from rag.category_tools.area_based import AreaBasedBillingTool, is_area_wound_question
from rag.category_tools.time_band import (
    TimeBandBillingTool,
    extract_discussion_minutes,
    is_category_g_question,
    is_phone_online_question,
    is_time_band_category_question,
)
from rag.category_tools.registry import get_tool_for_rule, list_category_tools
from rag.query_intent import UserIntent, classify_user_intent


@dataclass
class CategoryEngineOutcome:
    kind: Literal["answer", "clarification", "structured"]
    answer: str = ""
    clarification: str = ""
    structured: dict[str, Any] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    tool_name: str = ""
    billing_rule: str = ""
    category_id: str = ""

    @property
    def message(self) -> str:
        return self.clarification if self.kind == "clarification" else self.answer


_ADDON_HINTS = re.compile(r"\b(add[\s-]?on|addon|aoc|parent code)\b", re.IGNORECASE)
_RULE_HINTS = re.compile(
    r"\b(billing rule|under which rule|which rule|how (?:is|are) (?:it|units) "
    r"(?:billed|calculated)|category)\b",
    re.IGNORECASE,
)


def try_category_engine(question: str) -> CategoryEngineOutcome | None:
    """
    Run category-based billing before LLM/RAG when the question is rule-driven.
    Returns None when the category engine should not own the turn.
    """
    store = get_billing_category_store()
    store.ensure_loaded()

    intent = classify_user_intent(question)
    codes = _extract_billing_codes(question)
    entries = parse_cpt_time_entries(question)
    minutes_by_code = {e.cpt_code: e.minutes for e in entries}
    area_sq_cm = parse_area_sq_cm(question)
    methodology = detect_rule_methodology(question)

    # --- Category G: phone / online / explicit time_band_select (often no CPT yet) ---
    if is_category_g_question(question) or (
        not codes
        and extract_discussion_minutes(question) is not None
        and (
            is_time_band_category_question(question)
            or any(
                hint in question.lower()
                for hint in (
                    "advice",
                    "consult",
                    "spoke",
                    "called",
                    "discussion",
                    "session lasts",
                    "time band",
                    "time-band",
                    "time_band",
                )
            )
        )
    ):
        time_tool = TimeBandBillingTool()
        g_codes = [
            c
            for c in codes
            if (p := store.get_profile(c)) and p.billing_rule == TIME_BAND_SELECT
        ]
        g_profiles = [p for c in g_codes if (p := store.get_profile(c))]
        # Attach bare discussion minutes when no CPT-keyed time entries exist
        minutes = extract_discussion_minutes(question)
        if minutes is not None and not minutes_by_code:
            minutes_by_code = {"_discussion": minutes}
        result = time_tool.run(
            question=question,
            codes=g_codes or codes,
            profiles=g_profiles,
            minutes_by_code=minutes_by_code,
            area_sq_cm=area_sq_cm,
            methodology=methodology,
        )
        return _to_outcome(result, category_id="G")

    # --- Area-based wound scenarios (Category F) even without a CPT yet ---
    if is_area_wound_question(question) or (
        area_sq_cm is not None
        and intent.primary
        in {
            UserIntent.CODING_RECOMMENDATION,
            UserIntent.BILLING_UNIT_CALCULATION,
            UserIntent.UNKNOWN,
            UserIntent.GENERAL_HEALTHCARE_KNOWLEDGE,
        }
    ):
        area_tool = AreaBasedBillingTool()
        area_codes = [
            c
            for c in codes
            if (p := store.get_profile(c)) and p.billing_rule == AREA_BASED
        ]
        profiles = [p for c in area_codes if (p := store.get_profile(c))]
        result = area_tool.run(
            question=question,
            codes=area_codes or codes,
            profiles=profiles,
            minutes_by_code=minutes_by_code,
            area_sq_cm=area_sq_cm,
            methodology=methodology,
        )
        return _to_outcome(result, category_id="F")

    if not codes and not entries:
        return None

    profiles = []
    for code in codes or list(minutes_by_code):
        profile = store.get_profile(code)
        if profile is not None:
            profiles.append(profile)

    if not profiles:
        return None

    # --- Add-on validation questions ---
    if _ADDON_HINTS.search(question):
        tool = get_tool_for_rule("addon")
        if tool:
            result = tool.run(
                question=question,
                codes=codes,
                profiles=profiles,
                minutes_by_code=minutes_by_code,
                area_sq_cm=area_sq_cm,
                methodology=methodology,
            )
            return _to_outcome(result, category_id=profiles[0].category_id)

    # --- Unit calculation / guide / rule explanation ---
    wants_units = (
        is_unit_calculation_question(question)
        or _has_implicit_time_documentation(question)
        or is_unit_calculation_guide_question(question)
        or (entries and intent.primary == UserIntent.BILLING_UNIT_CALCULATION)
    )
    wants_rule = bool(_RULE_HINTS.search(question)) or intent.primary in {
        UserIntent.BILLING_UNIT_CALCULATION,
    }

    if not (wants_units or wants_rule or entries or area_sq_cm is not None):
        return None

    # Mixed-rule multi-CPT: compute each code via its own tool, then combine.
    # Category G (time_band_select) is CPT selection — never additive unit stacking.
    if len(profiles) > 1:
        return _run_mixed_profiles(
            question=question,
            profiles=profiles,
            minutes_by_code=minutes_by_code,
            area_sq_cm=area_sq_cm,
            methodology=methodology,
        )

    primary = profiles[0]
    tool = get_tool_for_rule(primary.billing_rule)
    if tool is None:
        return _category_summary(primary)

    # Prefer codes that share the primary rule.
    rule_profiles = [p for p in profiles if p.billing_rule == primary.billing_rule]
    result = tool.run(
        question=question,
        codes=[p.cpt_code for p in rule_profiles],
        profiles=rule_profiles,
        minutes_by_code=minutes_by_code,
        area_sq_cm=area_sq_cm,
        methodology=methodology,
    )
    return _to_outcome(result, category_id=primary.category_id)


def _run_mixed_profiles(
    *,
    question: str,
    profiles: list,
    minutes_by_code: dict[str, int],
    area_sq_cm: int | None,
    methodology: str,
) -> CategoryEngineOutcome:
    sections: list[str] = ["**Billing by CPT**", ""]
    structured_parts: list[dict[str, Any]] = []
    total = 0
    additive_codes = 0
    sources = ["billing_category_engine"]

    for profile in profiles:
        tool = get_tool_for_rule(profile.billing_rule)
        if tool is None:
            continue
        result = tool.run(
            question=question,
            codes=[profile.cpt_code],
            profiles=[profile],
            minutes_by_code=minutes_by_code,
            area_sq_cm=area_sq_cm,
            methodology=methodology,
        )
        if result.needs_clarification:
            return _to_outcome(result, category_id=profile.category_id)
        body = result.answer.replace("**Billing Unit Calculation**\n\n", "")
        title = f"### CPT {profile.cpt_code}"
        if profile.billing_rule == TIME_BAND_SELECT:
            title += " (time-band — select one CPT)"
        sections.append(title)
        sections.append(body)
        sections.append("")
        if profile.billing_rule == TIME_BAND_SELECT:
            # Category G: CPT selection only — do not add into unit totals.
            structured_parts.append(result.structured)
            sources.extend(result.sources)
            continue
        units = result.structured.get("units")
        if units is None:
            units = result.structured.get("total_units", 0)
        if isinstance(units, int):
            total += units
            additive_codes += 1
        else:
            match = re.search(r"Total Billing Units:\**\s*\**(\d+)", result.answer)
            if match:
                total += int(match.group(1))
                additive_codes += 1
        structured_parts.append(result.structured)
        sources.extend(result.sources)

    if additive_codes:
        sections.append(f"**Combined Total Billing Units (non–time-band CPTs):** **{total}**")
    return CategoryEngineOutcome(
        kind="answer",
        answer="\n".join(sections),
        structured={
            "mixed": True,
            "parts": structured_parts,
            "total_units": total,
            "processed_codes": [p.cpt_code for p in profiles],
        },
        sources=list(dict.fromkeys(sources)),
        tool_name="mixed_category_tools",
    )


def _category_summary(profile) -> CategoryEngineOutcome:
    store = get_billing_category_store()
    answer = (
        f"**Category {profile.category_id}** billing rules for "
        f"**{profile.cpt_code}**:\n\n"
        f"• **Billing rule:** `{profile.billing_rule}`\n"
        f"• **Definition:** {store.rule_definition(profile.billing_rule)}\n"
        f"• **Minimum requirement:** {profile.min_time_for_1_unit or 'See category'}\n"
        f"• **Max units allowed:** {profile.max_units_allowed or 'See category'}\n"
        f"• **Billing time:** {profile.billing_time or profile.time_band or profile.block_minutes or 'N/A'}"
    )
    return CategoryEngineOutcome(
        kind="answer",
        answer=answer,
        structured=store.to_structured_profile(profile.cpt_code) or {},
        sources=["billing_category_engine"],
        tool_name="category_summary",
        billing_rule=profile.billing_rule,
        category_id=profile.category_id,
    )


def _to_outcome(result, *, category_id: str = "") -> CategoryEngineOutcome:
    if result.needs_clarification:
        return CategoryEngineOutcome(
            kind="clarification",
            clarification=result.clarification or "",
            structured=result.structured,
            sources=result.sources,
            tool_name=result.tool_name,
            billing_rule=result.billing_rule,
            category_id=category_id,
        )
    return CategoryEngineOutcome(
        kind="answer",
        answer=result.answer,
        structured=result.structured,
        sources=result.sources,
        tool_name=result.tool_name,
        billing_rule=result.billing_rule,
        category_id=category_id,
    )


def describe_category_tooling() -> list[dict[str, str]]:
    """Expose tool catalog for future AWS Bedrock Agent registration."""
    return list_category_tools()
