"""Untimed category tools (session / encounter / procedure / day / episode)."""

from __future__ import annotations

from rag.billing_categories import CodeBillingProfile
from rag.billing_engine import (
    UNTIMED_RULES,
    CptTimeEntry,
    _format_standard_unit_answer,
    calculate_units_for_entry,
    mentions_same_day_multiple_occurrences,
)
from rag.category_tools import CategoryToolResult, profile_payload


class UntimedBillingTool:
    name = "untimed_billing_tool"
    billing_rules = tuple(sorted(UNTIMED_RULES))

    def supports(self, billing_rule: str) -> bool:
        return billing_rule in UNTIMED_RULES

    def run(
        self,
        *,
        question: str,
        codes: list[str],
        profiles: list[CodeBillingProfile],
        minutes_by_code: dict[str, int],
        area_sq_cm: int | None,
        methodology: str,
    ) -> CategoryToolResult:
        if not profiles:
            return CategoryToolResult(
                tool_name=self.name,
                billing_rule="untimed",
                answer="",
                needs_clarification=True,
                clarification="Which untimed CPT/HCPCS code are you billing?",
            )

        sections: list[str] = []
        structured_results: list[dict] = []
        total = 0

        for profile in profiles:
            minutes = minutes_by_code.get(profile.cpt_code, 0)
            entry = CptTimeEntry(cpt_code=profile.cpt_code, minutes=minutes)
            result = calculate_units_for_entry(
                entry, profile, methodology=methodology, area_sq_cm=area_sq_cm
            )
            total += result.units
            sections.append(
                _format_standard_unit_answer(
                    result, profile, include_title=len(profiles) == 1
                )
                if len(profiles) == 1
                else _format_standard_unit_answer(
                    result, profile, include_title=False
                )
            )
            structured_results.append(
                {
                    "profile": profile_payload(profile),
                    "units": result.units,
                    "minutes": minutes,
                }
            )

        if len(profiles) == 1:
            answer = sections[0]
        else:
            answer = "**Billing Unit Calculation**\n\n" + "\n\n".join(sections)
            answer += f"\n\n**Combined Total Billing Units:** **{total}**"

        if mentions_same_day_multiple_occurrences(question):
            capped = [
                p
                for p in profiles
                if p.max_units_cap == 1
                or "1 per day" in (p.max_units_allowed or "").lower()
            ]
            if capped:
                answer += (
                    "\n\nEven if performed more than once the same day, "
                    "billable units remain at the category maximum "
                    f"(**{capped[0].max_units_allowed}**)."
                )

        return CategoryToolResult(
            tool_name=self.name,
            billing_rule=profiles[0].billing_rule,
            answer=answer,
            structured={
                "tool": self.name,
                "total_units": total,
                "results": structured_results,
            },
        )
